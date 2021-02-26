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
"""ingle object detection model."""

import math

import tensorflow as tf


class SingleObjectModel:
  """Model using ResNet50 to predict single 3D oriented bounding box."""

  def __init__(self, is_training=False, output_dimension=30):
    super(SingleObjectModel, self).__init__()

    feature_backbone = tf.keras.applications.ResNet50(
        weights=None,
        input_shape=(256, 256, 3),
        include_top=False)
    feature_backbone.trainable = True
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = feature_backbone(inputs, training=is_training)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(output_dimension)(x)
    self.network = tf.keras.Model(inputs, outputs)
    self.huber_loss = tf.keras.losses.Huber(
        reduction=tf.keras.losses.Reduction.NONE)

  def compute_loss(self, inputs, outputs, global_batch_size):
    """Computes the losses and adds them to the output dict.

    Args:
      inputs: Dict of inputs to the model.
      outputs: Dict of ouputs of the model.
      global_batch_size: The batch size over all replicas.

    Returns:
      The updated outputs.
    """

    # Compute partial losses
    size2d_loss = self.huber_loss(outputs['size2d'], inputs['box_dim2d'])
    outputs['loss/size2d'] = size2d_loss
    center2d_loss = self.huber_loss(outputs['center2d'], inputs['center2d'])
    outputs['loss/center2d'] = center2d_loss

    size3d_loss = self.huber_loss(outputs['size3d'], inputs['box_dim3d'])
    outputs['loss/size3d'] = size3d_loss
    center3d_loss = self.huber_loss(outputs['center3d'], inputs['center3d'])
    outputs['loss/center3d'] = center3d_loss

    outputs['error/center3d_x'] = \
        tf.abs(outputs['center3d'][:, 0] - inputs['center3d'][:, 0])
    outputs['error/center3d_y'] = \
        tf.abs(outputs['center3d'][:, 1] - inputs['center3d'][:, 1])
    outputs['error/center3d_z'] = \
        tf.abs(outputs['center3d'][:, 2] - inputs['center3d'][:, 2])

    local_batch_size = outputs['size3d'].shape[0]
    rad = inputs['rotation'] * -1 * math.pi/180
    cos = tf.reshape(tf.cos(rad), [local_batch_size, 1, 1])
    sin = tf.reshape(tf.sin(rad), [local_batch_size, 1, 1])
    rotation = tf.concat([tf.concat([cos, sin], axis=2),
                          tf.concat([-sin, cos], axis=2)], axis=1)
    rotation_loss = tf.norm(outputs['rotation'] - rotation,
                            ord='fro', axis=[-2, -1])
    outputs['loss/rotation'] = rotation_loss

    # Compute mean over all losses based on global batch size
    for key in outputs.keys():
      if key.startswith('loss') or key.startswith('error'):
        outputs[key] = tf.nn.compute_average_loss(
            outputs[key], global_batch_size=global_batch_size)

    # Total loss
    total_loss = outputs['loss/center3d'] + \
        outputs['loss/size3d'] + outputs['loss/rotation']
    outputs['loss/total'] = total_loss
    return outputs

  def train_sample(self, sample, optimizer, global_batch_size):
    """Training step.

    Args:
      sample: Training sample.
      optimizer: The optimizer.
      global_batch_size: The global batch size.

    Returns:
      Dicts of model inputs and outputs.
    """

    inputs = {}
    self.decode_batch(sample, inputs)

    with tf.GradientTape() as t:
      predictions = self.network(inputs['image'])  # pytype: disable=key-error
      outputs = self.decode_predictions(predictions, inputs, {})
      outputs = self.compute_loss(inputs, outputs, global_batch_size)
      network_gradients = t.gradient(outputs['loss/total'],  # pytype: disable=key-error
                                     self.network.trainable_weights)
    optimizer.apply_gradients(zip(network_gradients,
                                  self.network.trainable_weights))
    return inputs, outputs

  @staticmethod
  def decode_batch(batch, inputs):
    """Decode batch and place into inputs dict.

    Args:
      batch: The input batch.
      inputs: The dict to put the decoded batch.

    Returns:
      Dict containing decoded input batch.
    """
    inputs['filename'] = batch[0]
    inputs['image'] = batch[1]
    inputs['image_size'] = batch[2]
    inputs['center2d'] = batch[3]
    inputs['center3d'] = batch[4]
    inputs['box_dim2d'] = batch[5]
    inputs['box_dim3d'] = batch[6]
    inputs['rotation'] = batch[7]
    inputs['rt'] = tf.reshape(batch[9], [-1, 3, 4])
    inputs['k'] = tf.reshape(batch[10], [-1, 3, 3])

  @staticmethod
  def decode_predictions(predictions, inputs, outputs):
    """Adds the decoded predictions to the output dict.

    Args:
      predictions: The predictions of the model.
      inputs: Input dict to the model.
      outputs: Output dict of the model.

    Returns:
      The updated output dict.
    """
    predicted_center2d_offset = predictions[:, 0:2]
    predicted_size2d_offset = predictions[:, 2:4]
    predicted_camera_pose = predictions[:, 4:9]  # 1d height, 2d rotation
    object_translation = predictions[:, 9:12]  # 3d translation
    object_rotation = predictions[:, 12:16]  # 4d rotation (rotation matrix)
    object_size = predictions[:, 16:19]  # 3d size

    # Camera Pose
    outputs['camera_translation_z'] = predicted_camera_pose[:, 0]
    outputs['camera_translation'] = 0
    outputs['camera_rotation_cos_sin'] = predicted_camera_pose[:, 1:]

    # Compute the dot on the ground plane used as prior for 3d pose (batched)
    image_size = inputs['image_size'][0]  # assuming it is the same size for all
    ray_2d = tf.cast(image_size, tf.float32) * [0.5, 3.0/4.0, 1/image_size[-1]]
    ray_2d = tf.reshape(ray_2d, [3, 1])
    k = inputs['k']
    k_inv = tf.linalg.inv(k)
    rt = inputs['rt']
    r = tf.gather(rt, [0, 1, 2], axis=2)
    t = tf.gather(rt, [3], axis=2)
    r_inv = tf.transpose(r, [0, 2, 1])
    t_inv = tf.matmul(r_inv, t) * -1
    ray = tf.matmul(r_inv, tf.matmul(k_inv, ray_2d))
    l = -t_inv[:, -1] / ray[:, -1]  # determine lambda
    dot = tf.expand_dims(l, -1) * ray + t_inv
    dot = tf.concat([dot, tf.ones([dot.shape[0], 1, 1])], axis=1)
    outputs['dot'] = dot

    # 3D Bounding Box: Size3D, Center3D (relative to dot), Rotation_z
    batch_size = dot.shape[0]
    size3d_prior = tf.cast([[1, 1, 1.5]], dtype=tf.float32)
    size3d_prior = tf.tile(size3d_prior, [batch_size, 1])
    outputs['size3d'] = size3d_prior + object_size
    center3d_prior = dot[:, 0:3, 0]
    outputs['center3d'] = center3d_prior + object_translation
    # Estimate rotation parameters using https://arxiv.org/abs/2006.14616
    rotation_prior = tf.eye(2, batch_shape=[batch_size])
    invalid_rotation = rotation_prior + tf.reshape(object_rotation, [-1, 2, 2])
    s, u, v = tf.linalg.svd(invalid_rotation)
    det = tf.linalg.det(tf.matmul(u, tf.linalg.matrix_transpose(v)))
    det = tf.reshape(det, [-1, 1])
    s = tf.linalg.diag(tf.concat([tf.ones(det.shape), det], axis=1))
    outputs['rotation'] = tf.matmul(u, tf.matmul(s, v, adjoint_b=True))

    # 2D Bounding Box: Center2D
    center2d_prior = tf.cast(inputs['image_size'][0, 0:2] / 2, dtype=tf.float32)
    predicted_center2d = center2d_prior + predicted_center2d_offset
    outputs['center2d'] = predicted_center2d

    # 2D Bounding Box: Size2D
    size2d_prior = tf.cast([100, 150], dtype=tf.float32)
    outputs['size2d'] = size2d_prior + predicted_size2d_offset

    return outputs
