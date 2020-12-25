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
"""Helper functions for medel definition."""

import numpy as np
import tensorflow.compat.v1 as tf


def transform_points(points, transform):
  """Transform points.

  Args:
    points: tf.Tensor, [batch_size, n_parts, n_points, n_dims], input points.
    transform: [batch_size, n_parts, transform_dims], transforming matices.

  Returns:
    points: tf.Tensor, [batch_size, n_parts, n_points, n_dims], output points.
  """
  batch_size = points.shape[0]
  n_parts = points.shape[1]
  n_points = tf.shape(points)[2]
  n_dims = points.shape[-1]
  ones = tf.ones_like(points[..., :1])
  points = tf.concat((points, ones), axis=-1)
  points = tf.reshape(points, (batch_size, n_parts, n_points, -1, 1))
  transform = tf.reshape(transform,
                         (batch_size, n_parts, 1, n_dims + 1, n_dims + 1))
  transform = tf.tile(transform, (1, 1, n_points, 1, 1))
  points = tf.matmul(transform, points)
  points = tf.squeeze(points, axis=-1)[..., :n_dims]
  return points


def compute_l2_indicator_loss(labels, predictions):
  return tf.losses.mean_squared_error(labels, predictions)


def get_identity_transform(n_translate, n_rotate, n_parts):
  """Generate the identity transformation as translation and rotation.

  Args:
    n_translate: int, the dimension of translation.
    n_rotate: int, the dimension of rotation.
    n_parts: int, the number of parts.

  Returns:
    transform: [1, 1, n_parts, n_translate + n_rotate], identity transformation.
  """
  translation = tf.zeros(n_translate)
  if n_rotate == 2:
    rotation = tf.convert_to_tensor((1., 0.), dtype=tf.float32)
  elif n_rotate == 6:
    rotation = tf.convert_to_tensor((
        1.,
        0.,
        0.,
        1.,
        0.,
        0.,
    ), dtype=tf.float32)
  else:
    raise ValueError("n_rotate should be either 6 or 2.")
  transform = tf.concat([translation, rotation], axis=-1)
  return tf.tile(
      tf.reshape(transform, [1, 1, 1, n_translate + n_rotate]),
      [1, 1, n_parts, 1])


def get_transform_matrix(transform, trans_range, n_translate, n_rotate, n_dims):
  """Generate the transformation matrix given translation and rotation.

  Args:
    transform: [batch_size, n_sample_frames, n_parts, transform_dims].
    [..., :n_translate] represents translations and
    [..., -n_rotat:] represents rotations.
    trans_range: float, the range of translation bound will be [-trans_range,
      trans_range].
    n_translate: int, the dimension of translation representations.
    n_rotate: int, the dimension of rotation representations.
    n_dims: int ,the dimension of points.

  Returns:
    matrix: tf.Tensor, [batch_size, n_frames, n_parts, (n_dims)^2].
  """
  batch_size = transform.shape[0]
  n_frames = transform.shape[1]
  n_parts = transform.shape[2]
  translation, rotation = tf.split(transform, [n_translate, n_rotate], axis=-1)
  translation = tf.tanh(translation) * trans_range  # Get bounded translation

  if n_rotate == 6:
    xy = tf.reshape(rotation, [batch_size, n_frames, n_parts, 3, 2])
    x, y = xy[..., :]
    x = tf.nn.l2_normalize(x, axis=-1)
    z = tf.linalg.cross(x, y)
    z = tf.nn.l2_normalize(z, axis=-1)
    y = tf.linalg.cross(z, x)
    rot_mat = tf.stack([x, y, z], axis=-1)
  elif n_rotate == 2:
    rotation = tf.nn.l2_normalize(rotation, axis=-1)
    rot_mat = tf.stack([
        rotation[..., 0], -rotation[..., 1], rotation[..., 1], rotation[..., 0]
    ],
                       axis=-1)
  else:
    raise ValueError("n_rotate should be either 3 or 2.")

  rot_mat = tf.reshape(rot_mat, [batch_size, n_frames, n_parts, n_dims, n_dims])
  translation = tf.reshape(translation,
                           [batch_size, n_frames, n_parts, n_dims, 1])
  ones = tf.ones_like(translation[..., :1, :])
  zeros = tf.reshape(
      tf.zeros_like(translation), [batch_size, n_frames, n_parts, 1, n_dims])
  up = tf.concat([rot_mat, translation], axis=-1)
  bottom = tf.concat([zeros, ones], axis=-1)
  trans_mat = tf.concat([up, bottom], axis=-2)
  return tf.reshape(trans_mat, [batch_size, n_frames, n_parts, (n_dims + 1)**2])


def nasa_indicator(points,
                   transform,
                   joint,
                   hparams,
                   need_transformation=True,
                   noise=None):
  """Compute the NASA indicator values for query points.

  Args:
    points: tf.Tensor, [batch, n_points, n_dims] or [batch_size, n_parts,
      n_points, n_dims], query points.
    transform: tf.Tensor, [batch, n_parts, transform_dims], transformations.
    joint: tf.Tensor, [batch_size, n_parts, n_dims], joint locations.
    hparams: hyperparameters as absl.flags.FLAGS.
    need_transformation: bool, true if points are transformed before fed to MLP.
    noise: tf.Tensor, [batch_size, n_parts, n_points, n_noise, n_dims], gaussian
      noise used to blur the boundary during tracking. Set to None if it is not
      needed.

  Returns:
    point_indicators: tf.Tensor, [batch, n_points, 1].
    point_perpart_indicators: tf.Tensor, [batch, n_parts, n_points, 1].
  """
  n_dims = 3
  cerb_dims = int(np.ceil(1. * hparams.total_dim / hparams.n_parts))
  shared_decoder = hparams.shared_decoder
  n_parts = hparams.n_parts
  use_joint = hparams.use_joint
  n_parts = hparams.n_parts
  batch_size = points.shape[0]

  if len(points.shape) < 4:
    points = tf.tile(tf.expand_dims(points, axis=1), [1, n_parts, 1, 1])
  if need_transformation:
    points = transform_points(points, transform)
  if noise is not None:
    points = tf.expand_dims(points, axis=3) + noise
    points = tf.reshape(points, [batch_size, n_parts, -1, n_dims])
  n_points = tf.shape(points)[2]

  if use_joint:
    root = tf.tile(tf.expand_dims(joint[:, :1], axis=1), [1, n_parts, 1, 1])
    root = transform_points(root, transform)
    if not hparams.projection:
      trans_feature = tf.tile(
          tf.reshape(root, [batch_size, 1, 1, -1]), [1, n_parts, n_points, 1])
    else:
      reduced_root = reduce_dimension(
          tf.reshape(root, [batch_size, -1]), n_parts)
      reduced_root = tf.reshape(reduced_root, [batch_size, n_parts, 1, -1])
      trans_feature = tf.tile(reduced_root, [1, 1, n_points, 1])
    points = tf.concat([points, trans_feature], axis=-1)

  return nasa_mlp(points, hparams, dims=cerb_dims, shared=shared_decoder)


def nasa_mlp(points, hparams, dims=256, shared=False):
  """Multi-layer perception for NASA.

  Args:
    points: tf.Tensor, [batch_size, n_parts, n_points, n_dims], query points.
    hparams: hyperparameters as absl.flags.FLAGS.
    dims: int, the dimension of hidden features within the MLP.
    shared: bool, true if share the MLP across branches.

  Returns:
    point_indicators: tf.Tensor, [batch, n_points, 1].
    point_perpart_indicators: tf.Tensor, [batch, n_parts, n_points, 1].
  """
  batch_size = points.shape[0]
  n_parts = points.shape[1]
  n_points = tf.shape(points)[2]
  n_point_dims = points.shape[-1]

  widths = [dims, dims, dims, dims, 1]
  n_widths = len(widths)
  indicators = []
  for idx in range(n_parts):
    x = points[:, idx]
    x = tf.reshape(x, [-1, n_point_dims])
    branch_id = 0 if shared else idx
    with tf.variable_scope("branch_{}".format(branch_id), reuse=tf.AUTO_REUSE):
      for i, dim in enumerate(widths):
        residual = x
        x = tf.layers.dense(
            inputs=x,
            units=dim,
            activation=None,
            name="fc_layer_{}".format(i),
        )
        if i == 0:
          x = tf.nn.leaky_relu(x, alpha=0.1)
        elif i < n_widths - 1:
          x = tf.nn.leaky_relu(x + residual, alpha=0.1)
    indicators.append(tf.reshape(x, [batch_size, n_points, 1]))
  x = tf.stack(indicators, axis=1)
  # Sigmoid Activation
  x = tf.nn.sigmoid(x)
  x_parts = x
  if hparams.soft_blend < 0.:
    # Max Blending
    x = tf.reduce_max(x, axis=1)
  else:
    # Soft-Max Blending
    weights = tf.nn.softmax(hparams.soft_blend * x, axis=1)
    x = tf.reduce_sum(weights * x, axis=1)
  return x, x_parts


def reduce_dimension(x, n_parts):
  x = tf.layers.dense(
      inputs=x,
      units=4 * n_parts,
      activation=None,
      use_bias=False,
      name="dim_reduction",
  )
  return x
