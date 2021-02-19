# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CenterNet network using Hourglass from https://arxiv.org/abs/1904.07850S."""
import collections
import os
import random
import string
import time

from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow_graphics.projects.points_to_3Dobjects.models import centernet_utils
from tensorflow_graphics.projects.points_to_3Dobjects.networks import hourglass
from tensorflow_graphics.projects.points_to_3Dobjects.transforms import transforms
from tensorflow_graphics.projects.points_to_3Dobjects.utils import image as image_utils

from google3.pyglib import gfile


class CenterNetVID:
  """CenterNet network using Hourglass from https://arxiv.org/abs/1904.07850S."""

  def __init__(self,
               network=hourglass.Hourglass,
               heads_losses=None,
               optimizer=None,
               get_k_predictions_test=3,
               score_threshold=0.0,
               pretrained_checkpoint_dir='',
               remove_pretrained_heads=False,
               input_shape=(None, 512, 512, 3),
               layers_to_train=None,
               clip_norm=None,
               shape_centers=None,
               shape_sdfs=None,
               beta=50,
               shape_pointclouds=None,
               dict_clusters=None,
               rotation_svd=True,
               **network_kwargs):
    self.k = get_k_predictions_test
    self.score_threshold = score_threshold
    self.sample_input_image = 'image'
    self.sample_image_size = 'original_image_spatial_shape'
    self.sample_num_boxes = 'num_boxes'
    self.shape_centers = shape_centers
    self.shape_sdfs = shape_sdfs
    self.shape_pointclouds = shape_pointclouds
    self.dict_clusters = dict_clusters
    self.rotation_svd = rotation_svd
    self.softargmax_sdf = centernet_utils.SoftArgMax(beta, shape_sdfs)
    self.heads_losses = heads_losses
    self.optimizer = optimizer
    self.network = network(input_shape, **network_kwargs)
    print('Building the network...')
    self.network.build(input_shape)
    print('Network built!')
    if pretrained_checkpoint_dir:
      if remove_pretrained_heads:
        tmp_network_kwargs = network_kwargs.copy()
        tmp_network_kwargs['heads'] = None
        network_no_heads = network(input_shape, **tmp_network_kwargs)
      else:
        network_no_heads = None
      self.load_ckpt(pretrained_checkpoint_dir, network_no_heads)

    if layers_to_train is not None:
      layers_to_train = [layers_to_train] if isinstance(
          layers_to_train, str) else layers_to_train
      for m in self.network.layers:
        if m.name in layers_to_train:
          m.trainable = True
        else:
          m.trainable = False
      self.network.summary()

    self.clip_norm = clip_norm
    self.beta = beta
    self.output_stride = self.network.output_stride

    # Parameters to set with the dataset
    self.batch_size = None
    self.window_size = None

  @staticmethod
  def _get_ckpt_dir(path):
    if gfile.IsDirectory(path):
      if path[-1] == '/':
        return path[:-1]
      else:
        return path
    else:
      return os.path.dirname(path)

  @staticmethod
  def _get_tmp_name(length=5):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase +
                                                string.digits)
                   for _ in range(length))

  def load_ckpt(self,
                checkpoint_dir_or_checkpoint_file_path,
                network_no_heads=None):
    """Load checkpoint."""

    def _load_ckpt(network, checkpoint_dir_or_checkpoint_file_path):
      """Load checkpoint helper.

      Args:
        network: Load weights in this network
        checkpoint_dir_or_checkpoint_file_path: Provide the path to a directory
          for checkpoints saved using save_weights(). Provide the path to a
          checkpoint file for checkpoints saved using tf.Checkpoint()
      Returns:
        A network
      """
      start = time.time()
      print(f'Loading pretrained checkpoint from '
            f'{checkpoint_dir_or_checkpoint_file_path}')
      try:
        if checkpoint_dir_or_checkpoint_file_path[-1] != '/':
          checkpoint_dir = f'{checkpoint_dir_or_checkpoint_file_path}/'
        else:
          checkpoint_dir = checkpoint_dir_or_checkpoint_file_path
        network.load_weights(checkpoint_dir)
      except tf.errors.NotFoundError:
        checkpoint = tf.train.Checkpoint(model=network)
        checkpoint.restore(checkpoint_dir_or_checkpoint_file_path)
      print(f'Checkpoint loaded in {time.time() - start} seconds.')
      return network

    if network_no_heads is not None:
      network_no_head = _load_ckpt(network_no_heads,
                                   checkpoint_dir_or_checkpoint_file_path)
      tmp_ckpt = tf.train.Checkpoint(model=network_no_head)
      tmp_ckpt_file = os.path.join(
          self._get_ckpt_dir(checkpoint_dir_or_checkpoint_file_path),
          self._get_tmp_name(), 'tmp')
      tmp_ckpt.save(tmp_ckpt_file)
      checkpoint_dir_or_checkpoint_file_path = f'{tmp_ckpt_file}-1'
    self.network = _load_ckpt(self.network,
                              checkpoint_dir_or_checkpoint_file_path)
    if network_no_heads is not None:
      gfile.DeleteRecursively(
          os.path.dirname(checkpoint_dir_or_checkpoint_file_path))
      del network_no_heads

  @staticmethod
  def _compute_loss(outputs, sample, losses, num_samples):
    """Loss helper."""
    per_output_per_head_loss = {}
    total_loss = 0
    for ii, output in enumerate([outputs[-1]]):
      for head_name in losses:
        head_output = output[head_name]
        head_loss = losses[head_name]['loss']
        if head_name in sample:
          head_gt = sample[head_name]
        else:
          head_gt = None
        head_weight = losses[head_name]['weight']
        current_loss = head_loss(head_output, head_gt, sample)
        info = f'Loss {head_name}:{current_loss}'
        logging.info(info)
        if not isinstance(current_loss, tf.Tensor):
          logging.info(current_loss.numpy())
        current_loss = tf.reduce_sum(current_loss) / tf.cast(
            num_samples, current_loss.dtype)
        per_output_per_head_loss[f'{head_name}_{ii}'] = current_loss
        total_loss = total_loss + current_loss * head_weight

    total_loss = total_loss / len(outputs)
    return total_loss, per_output_per_head_loss

  @staticmethod
  def merge_per_head_losses(total_loss, new_loss):
    for key in new_loss:
      total_loss[key] += new_loss[key]
    return total_loss

  @staticmethod
  def recover_sample_dimensions(outputs, sample_shape):
    for ii in range(len(outputs)):
      for key in outputs[ii]:
        output_shape = tf.shape(outputs[ii][key])
        outputs[ii][key] = tf.reshape(
            outputs[ii][key],
            tf.concat([sample_shape[:3], output_shape[1:]], axis=0))
    return outputs

  def compute_loss(self, sample, normalized_sample, outputs):
    """Compute loss."""
    if sample == 0:
      print('Nothing.')

    total_loss = 0
    per_output_per_head_loss = collections.defaultdict(float)

    if self.heads_losses is not None:
      image_loss, per_output_per_head_image_loss = self._compute_loss(
          outputs, normalized_sample, self.heads_losses, self.batch_size)
      total_loss += image_loss
      per_output_per_head_loss = self.merge_per_head_losses(
          per_output_per_head_loss, per_output_per_head_image_loss)

    if self.network.losses:
      regularization_loss = tf.reduce_mean(
          tf.stack(self.network.losses, axis=0))
      per_output_per_head_loss['regularization'] = regularization_loss
      total_loss += regularization_loss
    return total_loss, per_output_per_head_loss

  def more_things(self, output, sample, training=True, apply_sigmoid=True):
    """Helper function."""
    batch_size = output['centers'].shape[0]

    # Get shape-voxel-grid from shape-id.
    shape_logits = output['shapes']  # (BS, 128, 128, 300)
    indices = sample['indices']
    groundtruth_k = tf.shape(indices)[-1]

    if not training:
      centers = output['centers']
      if apply_sigmoid:
        centers = tf.math.sigmoid(centers)
        output['centers_sigmoid'] = centers

      centers = self.nms(centers)
      output['centers_nms'] = centers
      b, h, w, c = centers.shape
      assert b == 1
      centers = tf.reshape(centers, [1, h, w, c])
      _, _, _, _, _ = self._top_scores_heatmaps(
          centers, self.k)
      # offset = self._get_offsets(output['offset'], topk_inds)

    #   b, h, w, c = centers.shape
    # centers_t = tf.transpose(centers, [0, 3, 1, 2])
    # scores, indices = tf.math.top_k(tf.reshape(centers_t, [b, c, -1]), K)
    # topk_inds = indices % (h * w)
    # topk_ys = tf.cast(tf.cast((indices / w), tf.int32), tf.float32)
    # topk_xs = tf.cast(tf.cast((indices % w), tf.int32), tf.float32)
    # scores, indices = tf.math.top_k(tf.reshape(scores, [b, -1]), K)
    # topk_classes = tf.cast(indices / K, tf.int32)
    # topk_inds = tf.gather(
    #     tf.reshape(topk_inds, [b, -1]), indices, axis=1, batch_dims=1)
    # ys= tf.gather(tf.reshape(topk_ys, [b, -1]), indices, axis=1, batch_dims=1)
    # xs= tf.gather(tf.reshape(topk_xs, [b, -1]), indices, axis=1, batch_dims=1)
    # return xs, ys, topk_classes, topk_inds, scores

    # indices = topk_inds
    top_shape_logits = centernet_utils.get_heatmap_values(shape_logits, indices)

    sizes_3d, translations_3d, rotations_3d = \
        centernet_utils.decode_box_3d(output, indices, self.rotation_svd)

    # b, h, w, c = centers.shape
    # centers_2d_indices = indices % (h * w)
    # ys_2d = tf.cast(tf.cast((centers_2d_indices / w), tf.int32), tf.float32)
    # xs_2d = tf.cast(tf.cast((centers_2d_indices % w), tf.int32), tf.float32)

    # center2d = sample['center2d']
    # offset_proj_3dS = sample['offset']
    # xs, ys = xs + offset[..., 0], ys + offset[..., 1]
    #
    # center_2d = [xs_2d, ys_2d] * self.output_stride
    #
    # offset_x = translations_3d[:, 0]
    # offset_y = translations_3d[:, 1]
    # depth = translations_3d[:, 2]

    output['sizes_3d'] = sizes_3d
    output['translations_3d'] = translations_3d
    output['rotations_3d'] = rotations_3d

    # Get ground truth point cloud
    groundtruth_pointclouds = tf.gather(self.shape_pointclouds,
                                        tf.expand_dims(sample['shapes'],
                                                       axis=-1))
    groundtruth_pointclouds = tf.reshape(groundtruth_pointclouds,
                                         [batch_size, -1, 512, 3])

    # Transform ground truth point cloud using ground truth pose
    groundtruth_pointclouds_groundtruth_transformed = \
        centernet_utils.transform_pointcloud(
            groundtruth_pointclouds / 2.0,
            sample['sizes_3d'],
            sample['rotations_3d'],
            sample['translations_3d'])
    sample['pose_groundtruth_pointclouds'] = \
        groundtruth_pointclouds_groundtruth_transformed

    # Transform ground truth point cloud using predicted pose
    groundtruth_pointclouds_predicted_transformed = \
        centernet_utils.transform_pointcloud(
            groundtruth_pointclouds / 2.0,
            sample['sizes_3d'],
            output['rotations_3d'],
            output['translations_3d'])
    output['pose_groundtruth_pointclouds'] = \
        groundtruth_pointclouds_predicted_transformed

    # Get predicted SDFs
    predicted_sdfs = centernet_utils.softargmax(top_shape_logits,
                                                self.shape_sdfs,
                                                self.beta)
    # predicted_sdfs = self.softargmax_sdf(top_shape_logits)
    predicted_sdfs = tf.reshape(predicted_sdfs, [batch_size, -1, 32, 32, 32])
    output['sdfs'] = predicted_sdfs

    # Get predicted pointclouds
    predicted_pointclouds = centernet_utils.softargmax(top_shape_logits,
                                                       self.shape_pointclouds,
                                                       self.beta)
    predicted_pointclouds = tf.reshape(predicted_pointclouds,
                                       [batch_size, -1, 512, 3])

    groundtruth_sdfs = tf.squeeze(tf.gather(self.shape_sdfs,
                                            tf.expand_dims(sample['shapes'],
                                                           axis=-1)),
                                  axis=2)

    output['collisions'] = (predicted_sdfs, predicted_pointclouds, sizes_3d,
                            translations_3d, rotations_3d)
    output['collisions_gt_shapes'] = (groundtruth_sdfs, groundtruth_pointclouds,
                                      sample['sizes_3d'],
                                      translations_3d, rotations_3d)

    # predicted_pointclouds =
    # (predicted_pointclouds *  (29.0/32.0) / 2.0 + 0.5) * 32.0 - 0.5
    # sdf_values = trilinear.interpolate(tf.expand_dims(predicted_sdfs, -1),
    # predicted_pointclouds)

    output['groundtruth_sdfs'] = groundtruth_sdfs
    output['predicted_sdfs'] = predicted_sdfs
    output['groundtruth_pointclouds'] = groundtruth_pointclouds
    output['predicted_pointclouds'] = predicted_pointclouds

    output['sdfs'] = tf.reshape(predicted_sdfs, [batch_size, -1, 32**3, 1])
    sample['sdfs'] = tf.reshape(groundtruth_sdfs, [batch_size, -1, 32**3, 1])

    # Transform predicted point cloud
    transformed_pointclouds = \
        centernet_utils.transform_pointcloud(predicted_pointclouds / 2.0,
                                             sizes_3d, rotations_3d,
                                             translations_3d)
    transformed_pointclouds = \
        tf.concat([transformed_pointclouds,
                   tf.ones([batch_size, groundtruth_k, 512, 1])], axis=-1)
    transformed_pointclouds = tf.transpose(transformed_pointclouds,
                                           [0, 1, 3, 2])  # (5, 3, 4, 512)
    output['transformed_pointclouds'] = transformed_pointclouds

    intrinsics = tf.reshape(sample['k'], [batch_size, 1, 3, 3])  # (5, 3, 3)
    intrinsics = tf.tile(intrinsics, [1, groundtruth_k, 1, 1])  # (5, 1, 3, 4)
    extrinsics = tf.expand_dims(sample['rt'], axis=1)  # (5, 1, 3, 4)
    extrinsics = tf.tile(extrinsics, [1, groundtruth_k, 1, 1])  # (5, 1, 3, 4)

    projected_pointclouds = intrinsics @ extrinsics @ transformed_pointclouds
    projected_pointclouds = tf.transpose(projected_pointclouds, [0, 1, 3, 2])
    projected_pointclouds = \
        projected_pointclouds / projected_pointclouds[:, :, :, -1:]
    output['projected_pointclouds'] = projected_pointclouds

    # 2D Loss preparation
    pointcloud = groundtruth_pointclouds_groundtruth_transformed
    pointcloud = tf.concat([pointcloud,
                            tf.ones([batch_size, groundtruth_k, 512, 1])],
                           axis=-1)
    pointcloud = tf.transpose(pointcloud, [0, 1, 3, 2])  # (5, 3, 4, 512)
    pointcloud = intrinsics @ extrinsics @ pointcloud
    pointcloud = tf.transpose(pointcloud, [0, 1, 3, 2])
    sample['projected_gt_shapes'] = \
        (pointcloud / pointcloud[:, :, :, -1:])[:, :, :, 0:2]

    pointcloud = groundtruth_pointclouds_predicted_transformed
    pointcloud = tf.concat([pointcloud,
                            tf.ones([batch_size, groundtruth_k, 512, 1])],
                           axis=-1)
    pointcloud = tf.transpose(pointcloud, [0, 1, 3, 2])  # (5, 3, 4, 512)
    pointcloud = intrinsics @ extrinsics @ pointcloud
    pointcloud = tf.transpose(pointcloud, [0, 1, 3, 2])
    output['projected_gt_shapes'] = \
        (pointcloud / pointcloud[:, :, :, -1:])[:, :, :, 0:2]

    return output

  def test_sample(self, sample, compute_loss=False, training=False):
    """Test function."""
    outputs = self.network(sample[self.sample_input_image], training=training)
    outputs[-1] = self.more_things(outputs[-1], sample, training=False)

    loss = None
    if compute_loss:
      total_loss, per_output_per_head_loss = self.compute_loss(
          sample, sample, outputs)
      loss = {'total_loss': total_loss, **per_output_per_head_loss}
    return outputs, loss

  def train_sample(self, sample):
    """Train function."""
    with tf.GradientTape() as grad_tape:
      outputs = self.network(sample[self.sample_input_image], training=True)
      outputs[-1] = self.more_things(outputs[-1], sample, training=True)
      total_loss, per_output_per_head_loss = self.compute_loss(
          sample, sample, outputs)
    network_gradients = grad_tape.gradient(total_loss,
                                           self.network.trainable_variables)
    for i, grad in enumerate(network_gradients):
      if grad is not None:
        tf.debugging.check_numerics(grad, 'Invalid gradients: '+str(i))

    gradients_norm = tf.constant(0)
    if self.clip_norm:
      network_gradients, gradients_norm = tf.clip_by_global_norm(
          network_gradients, clip_norm=self.clip_norm)

    self.optimizer.apply_gradients(
        zip(network_gradients, self.network.trainable_variables))

    additional_logs = {'total_loss': total_loss,
                       'gradients_norm': gradients_norm}

    return {** additional_logs, **per_output_per_head_loss}

  @staticmethod
  def nms(centers):
    centers_max = tf.nn.max_pool2d(centers, 3, 1, 'SAME')
    centers_keep = tf.cast(tf.abs(centers_max - centers) < 1e-6, tf.float32)
    centers_nms = tf.multiply(centers_keep, centers)
    return centers_nms

  @staticmethod
  def _top_scores_heatmaps(centers, k):
    """Get top scores from heatmaps."""
    b, h, w, c = centers.shape
    centers_t = tf.transpose(centers, [0, 3, 1, 2])
    scores, indices = tf.math.top_k(tf.reshape(centers_t, [b, c, -1]), k)
    topk_inds = indices % (h * w)
    topk_ys = tf.cast(tf.cast((indices / w), tf.int32), tf.float32)
    topk_xs = tf.cast(tf.cast((indices % w), tf.int32), tf.float32)
    scores, indices = tf.math.top_k(tf.reshape(scores, [b, -1]), k)
    topk_classes = tf.cast(indices / k, tf.int32)
    topk_inds = tf.gather(
        tf.reshape(topk_inds, [b, -1]), indices, axis=1, batch_dims=1)
    ys = tf.gather(tf.reshape(topk_ys, [b, -1]), indices, axis=1, batch_dims=1)
    xs = tf.gather(tf.reshape(topk_xs, [b, -1]), indices, axis=1, batch_dims=1)

    return xs, ys, topk_classes, topk_inds, scores

  @staticmethod
  def _get_offsets(offset, topk_inds):
    b, _, _, n = offset.shape
    offset = tf.reshape(offset, [b, -1, n])
    offset = tf.gather(offset, topk_inds, batch_dims=1)
    return offset

  @staticmethod
  def _get_width_height(width_height, topk_inds):
    b, _, _, n = width_height.shape
    width_height = tf.reshape(width_height, [b, -1, n])
    width_height = tf.gather(width_height, topk_inds, batch_dims=1)
    return width_height

  @staticmethod
  def group_detections(xs, ys, scores, classes, width_height):
    """Group detections."""
    classes = tf.cast(classes[..., None], tf.float32)
    scores = scores[..., None]
    xs, ys = xs[..., None], ys[..., None]
    bboxes = tf.concat([
        xs - width_height[..., 0:1] / 2, ys - width_height[..., 1:2] / 2,
        xs + width_height[..., 0:1] / 2, ys + width_height[..., 1:2] / 2
    ],
                       axis=-1)
    detections = tf.concat([bboxes, scores, classes], axis=-1)
    return detections

  @staticmethod
  def transform_and_group_detections_per_class(detections, metadata,
                                               num_classes):
    """Transform and group detections."""
    c, s = metadata['center'], metadata['side_size']
    w, h = metadata['output_w'], metadata['output_h']
    detections = detections.numpy()
    ret = []
    for i in range(detections.shape[0]):
      top_preds = {}
      detections[i, :, :2] = image_utils.transform_predictions(
          detections[i, :, 0:2], c, s, (w, h))
      detections[i, :, 2:4] = image_utils.transform_predictions(
          detections[i, :, 2:4], c, s, (w, h))
      classes = detections[i, :, -1]
      for j in range(num_classes):
        inds = (classes == j)
        top_preds[j + 1] = np.concatenate([
            detections[i, inds, :4].astype(np.float32),
            detections[i, inds, 4:5].astype(np.float32)
        ],
                                          axis=1)
      ret.append(top_preds)
    return ret

  def transform_detections(self, detections, metadata):
    """Transform detections."""
    all_detections = []
    batch_size, _, _ = detections.shape
    for ii in range(batch_size):
      metadata_ii = metadata[ii, ...] if tf.rank(metadata) == 3 else metadata
      output_size = metadata_ii[2, :] // self.output_stride
      points = tf.reshape(detections[ii, :, :4], [-1, 2])
      new_points = transforms.transform_predictions(points, metadata_ii[0, :],
                                                    metadata_ii[1, :],
                                                    output_size)
      detection = tf.concat(
          [tf.reshape(new_points, [-1, 4]), detections[ii, :, 4:]], axis=1)
      all_detections.append(detection)
    return tf.stack(all_detections, axis=0)

  def postprocess_sample2(self, input_image_size, sample, output,
                          apply_sigmoid=True):
    """Post-process model predictions into detections."""

    if apply_sigmoid:
      print(apply_sigmoid)

    batch_id = 0
    detections_dict = {}
    detections_dict['centers'] = output['centers'][batch_id, ...]
    detections_dict['centers_sigmoid'] = \
        output['centers_sigmoid'][batch_id, ...]
    detections_dict['centers_nms'] = output['centers_nms'][batch_id, ...]

    # self.K = sample['num_boxes'][0].numpy()

    xs, ys, topk_classes, topk_inds, scores = self._top_scores_heatmaps(
        output['centers_nms'], self.k)

    # Filter based on threshold
    mask = scores > self.score_threshold
    xs = tf.expand_dims(xs[mask], 0)
    ys = tf.expand_dims(ys[mask], 0)
    topk_classes = tf.expand_dims(topk_classes[mask], 0)
    topk_inds = tf.expand_dims(topk_inds[mask], 0)
    scores = tf.expand_dims(scores[mask], 0)
    k = tf.shape(scores)[-1]

    offset = self._get_offsets(output['offset'], topk_inds)
    xs, ys = xs + offset[..., 0], ys + offset[..., 1]
    width_height = self._get_width_height(output['width_height'], topk_inds)
    detections = self.group_detections(xs, ys, scores, topk_classes,
                                       width_height)
    original_image_size = sample[self.sample_image_size][batch_id, :2]
    center, side_size, input_size = transforms. \
      compute_image_size_affine_transform(original_image_size, input_image_size)
    metadata = tf.stack([center, side_size, input_size], axis=0)[None, ...]
    detections = self.transform_detections(detections, metadata)

    sizes_3d, translations_3d, rotations_3d = \
        centernet_utils.decode_box_3d(output, topk_inds, self.rotation_svd)
    rotations_3d = tf.reshape(rotations_3d, [-1, k, 3, 3])

    # Shape post-processing
    shape_logits = self._get_width_height(output['shapes'], topk_inds)

    status = True
    if status:
      detection_classes = tf.cast(detections[:, :, -1], dtype=tf.int32)
      # 3   8   0- 50  bottle
      # 4  56  50-100  bowl
      # 0 119 100-150  chair
      # 5 156 150-200  mug
      # 1 202 200-250  sofa
      # 2 287 250-300  table

      num_classes = 6
      cluster_size = 50
      detection_classes_reorder = \
          np.array([3, 4, 0, 5, 1, 2])[detection_classes.numpy()]
      mask = tf.repeat(tf.one_hot(detection_classes_reorder, num_classes,
                                  axis=-1),
                       repeats=[cluster_size], axis=-1)
      shape_logits = shape_logits - (1 - mask) * tf.reduce_min(shape_logits,
                                                               axis=-1,
                                                               keepdims=True)

    shape_classes = tf.argmax(shape_logits, axis=-1)

    # Get predicted SDFs and Pointclouds
    predicted_sdfs = centernet_utils.softargmax(shape_logits, self.shape_sdfs)
    predicted_sdfs = tf.reshape(predicted_sdfs, [1, -1, 32, 32, 32])
    predicted_pointclouds = centernet_utils.softargmax(shape_logits,
                                                       self.shape_pointclouds)
    predicted_pointclouds = tf.reshape(predicted_pointclouds, [1, -1, 512, 3])

    # Transform predicted point cloud
    transformed_pointclouds = \
        centernet_utils.transform_pointcloud(predicted_pointclouds / 2.0,
                                             sizes_3d,
                                             rotations_3d,
                                             translations_3d)
    transformed_pointclouds = tf.concat([transformed_pointclouds,
                                         tf.ones([1, k, 512, 1])], axis=-1)
    transformed_pointclouds = tf.transpose(transformed_pointclouds,
                                           [0, 1, 3, 2])  # (5, 3, 4, 512)

    intrinsics = tf.reshape(sample['k'], [1, 1, 3, 3])  # (5, 3, 3)
    intrinsics = tf.tile(intrinsics, [1, k, 1, 1])  # (5, 1, 3, 4)
    extrinsics = tf.expand_dims(sample['rt'], axis=1)  # (5, 1, 3, 4)
    extrinsics = tf.tile(extrinsics, [1, k, 1, 1])  # (5, 1, 3, 4)

    projected_pointclouds = intrinsics @ extrinsics @ transformed_pointclouds
    projected_pointclouds = tf.transpose(projected_pointclouds, [0, 1, 3, 2])
    projected_pointclouds = \
        projected_pointclouds / projected_pointclouds[:, :, :, -1:]

    # detections_dict['centers'] = detections_dict['centers']
    # detections_dict['centers_sigmoid'] = detections_dict['centers_sigmoid']
    # detections_dict['centers_nms'] = detections_dict['centers_nms']
    detections_dict['detection_boxes'] = detections[batch_id, :, :4]
    detections_dict['detection_scores'] = detections[batch_id, :, 4]
    detections_dict['detection_classes'] = detections[batch_id, :, 5]
    detections_dict['rotations_3d'] = rotations_3d[batch_id, ...]
    detections_dict['translations_3d'] = translations_3d[batch_id, ...]
    detections_dict['sizes_3d'] = sizes_3d[batch_id, ...]
    detections_dict['groundtruth_sdfs'] = \
        output['groundtruth_sdfs'][batch_id, ...]
    detections_dict['groundtruth_pointclouds'] = \
        output['groundtruth_pointclouds'][batch_id, ...]
    detections_dict['predicted_sdfs'] = predicted_sdfs[batch_id, ...]
    detections_dict['predicted_pointclouds'] = \
        predicted_pointclouds[batch_id, ...]
    detections_dict['shapes'] = shape_classes[batch_id, ...]
    detections_dict['shapes_logits'] = shape_logits[batch_id, ...]
    detections_dict['projected_pointclouds'] = \
        projected_pointclouds[batch_id, ...]
    return detections_dict
