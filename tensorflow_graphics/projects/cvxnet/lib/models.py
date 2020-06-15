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
"""Model Implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.cvxnet.lib import resnet

keras = tf.keras


def get_model(model_name, args):
  return model_dict[model_name](args)


class MultiConvexNet(keras.Model):
  """Shape auto-encoder with multiple convex polytopes.

  Attributes:
    n_params: int, the number of hyperplane parameters.
  """

  def __init__(self, args):
    super(MultiConvexNet, self).__init__()
    self._k = 20
    self._n_half_planes = args.n_half_planes
    self._n_parts = args.n_parts
    self._sample_bbx = args.sample_bbx
    self._sample_surf = args.sample_surf
    self._sample_size = args.sample_bbx + args.sample_surf
    self._level_set = args.level_set
    self._w_overlap = args.weight_overlap
    self._w_balance = args.weight_balance
    self._w_center = args.weight_center
    self._image_input = args.image_input
    self._dims = args.dims

    # Params = Roundness + Translation + Hyperplanes
    self.n_params = self._n_parts + \
                    self._n_parts * self._dims + \
                    self._n_parts * self._n_half_planes * self._dims

    with tf.variable_scope("mc_encoder"):
      self.img_encoder = resnet.Resnet18(args.latent_size)
      self.beta_decoder = Decoder(self.n_params)

    with tf.variable_scope("mc_decoder"):
      self.sdf = MultiConvexUnit(args.dims, args.n_parts, args.sharpness)

  def compute_loss(self, batch, training, optimizer=None):
    """Compute loss given a batch of data.

    Args:
      batch: Dict, must contains:
        "image": [batch_size, image_h, image_w, image_d],
        "depth": [batch_size, depth_h, depth_w, depth_d],
        "point": [batch_size, n_points, dims],
        "point_label": [batch_size, n_points, 1],
      training: bool, use training mode if true.
      optimizer: tf.train.Optimizer, optimizer used for training.

    Returns:
      train_loss: tf.Operation, loss hook.
      train_op: tf.Operation, optimization op.
      global_step: tf.Operation, gloabl step hook.
    """
    img = batch["image"]
    depth = batch["depth"]
    points = batch["point"]
    gt = batch["point_label"]

    if self._image_input:
      beta = self.encode(img, training=training)
    else:
      beta = self.encode(depth, training=training)
    output, (trans, imgsum, offset, image_indica) = self.decode(
        beta, points, training=training)

    sample_loss = self._compute_sample_loss(gt, output)

    overlap_loss = self._compute_overlap_loss(imgsum)

    bbx_loss = self._compute_bbx_loss(trans, points, gt)

    center_loss = tf.reduce_mean(tf.square(offset))

    balance_loss = self._compute_balance_loss(image_indica, gt)

    loss = (
        overlap_loss * self._w_overlap + balance_loss * self._w_balance +
        center_loss * self._w_center + bbx_loss + sample_loss)

    if training:
      tf.summary.scalar("sample_loss", sample_loss)
      tf.summary.scalar("overlap_loss", overlap_loss)
      tf.summary.scalar("bbx_loss", bbx_loss)
      tf.summary.scalar("center_loss", center_loss)
      tf.summary.scalar("balance_loss", balance_loss)
      tf.summary.scalar("train_loss", loss)
      tf.summary.image("input_images", img)

    if training:
      global_step = tf.train.get_or_create_global_step()
      update_ops = self.updates
      with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, unused_var = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)
      return loss, train_op, global_step
    else:
      occ = tf.cast(output >= self._level_set, tf.float32)
      intersection = tf.reduce_sum(occ * gt, axis=(1, 2))
      union = tf.reduce_sum(tf.maximum(occ, gt), axis=(1, 2))
      iou = intersection / union
      return loss, iou

  def encode(self, img, training):
    """Encode the input image into hyperplane parameters.

    Args:
      img: Tensor, [batch_size, height, width, channels], input image.
      training: bool, use training mode if true.

    Returns:
      params: Tensor, [batch_size, n_params], hyperplane parameters.
    """
    x = self.img_encoder(img, training=training)
    return self.beta_decoder(x, training=training)

  def decode(self, params, points, training):
    """Decode the hyperplane parameters into indicator fuctions.

    Args:
      params: Tensor, [batch_size, n_params], hyperplane parameters.
      points: Tensor, [batch_size, n_points, dims], query points.
      training: bool, use training mode if true.

    Returns:
      indicator: Tensor, [batch_size, n_points, 1], indicators for query points.
      extra: list, contains:
        trans: Tensor, [batch_size, n_parts, dims], translations.
        imgsum: Tensor, [batch_size, n_points, 1], sum of indicators.
        offset: Tensor, [batch_size, n_parts, n_half_planes, 1], off set of
        hyperplanes.
        image_indica: Tensor, [batch_Size, n_parts, n_points, 1], per part
        indicators.
    """
    theta, translations, blend_terms = self._split_params(params)
    indicator, (trans, imgsum, offset,
                image_indica) = self.sdf(theta, translations, blend_terms,
                                         points)
    return indicator, (trans, imgsum, offset, image_indica)

  def _split_params(self, params):
    """Split the parameter tensor."""
    blend_terms, translations, theta = tf.split(
        params, [
            self._n_parts, self._n_parts * self._dims,
            self._n_parts * self._n_half_planes * self._dims
        ],
        axis=-1)
    translations = tf.reshape(translations, [-1, self._n_parts, self._dims])
    theta = tf.reshape(theta,
                       [-1, self._n_parts, self._n_half_planes, self._dims])
    return theta, translations, blend_terms

  def _compute_sample_loss(self, gt, output):
    """Compute point sampling loss."""
    sample_loss = tf.square(gt - output)
    if self._sample_bbx > 0:
      loss_bbx = sample_loss[:, :self._sample_bbx]
      loss_bbx = tf.reduce_mean(loss_bbx)
    else:
      loss_bbx = 0.
    if self._sample_surf > 0:
      loss_surf = sample_loss[:, self._sample_bbx:]
      loss_surf = tf.reduce_mean(loss_surf)
    else:
      loss_surf = 0.
    sample_loss = loss_bbx + 0.1 * loss_surf
    return sample_loss

  def _compute_bbx_loss(self, trans, pts, gt):
    """Compute bounding box loss."""
    oo = 1e5
    inside = tf.expand_dims(tf.cast(gt > 0.5, tf.float32), axis=1)
    trans = tf.expand_dims(trans, axis=2)
    pts = tf.expand_dims(pts, axis=1)
    distances = tf.reduce_sum(tf.square(trans - pts), axis=-1, keepdims=True)
    distances = inside * distances + (1 - inside) * oo
    min_dis = tf.reduce_min(distances, axis=2)
    return tf.reduce_mean(min_dis)

  def _compute_overlap_loss(self, overlap):
    """Compute overlap loss."""
    return tf.reduce_mean(tf.square(tf.nn.relu(overlap - 2.)))

  def _compute_balance_loss(self, indicas, labels):
    """Compute balance loss."""
    indicas = indicas * tf.expand_dims(labels, axis=1)
    indicas = tf.squeeze(indicas, axis=-1)  # Have to squeeze it for top_k
    vals, unused_var = tf.nn.top_k(indicas, k=self._k)
    return tf.reduce_mean(tf.square(1. - vals))


class MultiConvexUnit(keras.layers.Layer):
  """Differentiable shape rendering layer using multiple convex polytopes."""

  def __init__(self, dims, n_parts, sharpness=75.):
    super(MultiConvexUnit, self).__init__()
    self._offset_scale = 0.5
    self._offset_lbound = 0
    self._blend_scale = 250.
    self._blend_lbound = 50.
    self._sharpness = sharpness
    self._dims = dims
    self._n_parts = n_parts

  def call(self, x, translations, blend_terms, points):
    """Construct object by assembling convex polytopes differentiably.

    Args:
      x: Tensor, [batch_size, n_parts, n_half_planes, dims], hyperplane
        parameters.
      translations: Tensor, [batch_size, n_parts, dims], translation vectors.
      blend_terms: Tensor, [batch_size, n_parts], smoothness terms for blending
        hyperplanes.
      points: Tensor, [batch_size, n_points, dims], query points.

    Returns:
      indicator: Tensor, [batch_size, n_points, 1], indicators for query points.
      extra: list, contains:
        trans: Tensor, [batch_size, n_parts, dims], translations.
        imgsum: Tensor, [batch_size, n_points, 1], sum of indicators.
        offset: Tensor, [batch_size, n_parts, n_half_planes, 1], offset of
        hyperplanes.
        image_indica: Tensor, [batch_Size, n_parts, n_points, 1], per part
        indicators.
    """
    points = tf.concat([points, translations], axis=1)
    signed_dis, transform, blend_planes, offset = self._compute_sdf(
        x, translations, blend_terms, points)

    # Generate convex shapes (use logsumexp as the intersection of half-spaces)
    part_logits = tf.reduce_logsumexp(
        signed_dis * tf.reshape(blend_planes, [-1, self._n_parts, 1, 1, 1]),
        axis=2)
    part_logits = (-part_logits /
                   tf.reshape(blend_planes, [-1, self._n_parts, 1, 1]))

    part_indica_full = tf.nn.sigmoid(part_logits * self._sharpness)
    part_indica = part_indica_full[:, :, :-self._n_parts]

    image_indica_sum = tf.reduce_sum(part_indica_full, axis=1)
    image_indica_max = tf.reduce_max(part_indica, axis=1)

    return image_indica_max, (transform, image_indica_sum, offset, part_indica)

  def _compute_sdf(self, x, translations, blend_terms, points):
    """Compute signed distances between query points and hyperplanes."""
    n_parts = tf.shape(x)[1]
    n_planes = tf.shape(x)[2]
    norm_logit = x[..., 0:self._dims - 1]
    offset = (-(tf.nn.sigmoid(x[..., self._dims - 1:self._dims]) *
                self._offset_scale + self._offset_lbound))
    blend_planes = (
        tf.nn.sigmoid(blend_terms[..., :n_parts]) * self._blend_scale +
        self._blend_lbound)

    # Norm of the boundary line
    norm_rad = tf.tanh(norm_logit) * np.pi  # [..., (azimuth, altitude)]
    if self._dims == 3:
      norm = tf.stack([
          tf.sin(norm_rad[..., 1]) * tf.cos(norm_rad[..., 0]),
          tf.sin(norm_rad[..., 1]) * tf.sin(norm_rad[..., 0]),
          tf.cos(norm_rad[..., 1])
      ],
                      axis=-1)
    else:
      norm = tf.concat([tf.cos(norm_rad), tf.sin(norm_rad)], axis=-1)

    # Calculate signed distances to hyperplanes.
    points = (
        tf.expand_dims(points, axis=1) - tf.expand_dims(translations, axis=2))
    points = tf.expand_dims(points, axis=2)
    points = tf.tile(points, [1, 1, n_planes, 1, 1])
    signed_dis = tf.matmul(points, tf.expand_dims(norm, axis=-1))
    signed_dis = signed_dis + tf.expand_dims(offset, axis=-2)

    return signed_dis, translations, blend_planes, offset


class Decoder(keras.layers.Layer):
  """MLP decoder to decode latent codes to hyperplane parameters."""

  def __init__(self, dims):
    super(Decoder, self).__init__()
    self._decoder = keras.Sequential()
    layer_sizes = [1024, 1024, 2048]
    for layer_size in layer_sizes:
      self._decoder.add(
          keras.layers.Dense(layer_size, activation=tf.nn.leaky_relu))
    self._decoder.add(keras.layers.Dense(dims, activation=None))

  def call(self, x, training):
    return self._decoder(x, training=training)


model_dict = {
    "multiconvex": MultiConvexNet,
}
