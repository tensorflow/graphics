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
"""Utility methods for point cloud layers."""

import tensorflow as tf
import numpy as np

tf_pi = tf.convert_to_tensor(np.pi)


def _format_output(features, point_cloud, return_sorted, return_padded):
  """ Method to format and sort the output of a point cloud convolution layer.

  Note:
    In the following, `A1` to `An` are optional batch dimensions.

  Args:
    features: A `float` `Tensor`  of shape `[N, C]`.
    point_cloud: A `PointCloud` instance, on which the `feautres` are defined.
    return_sorted: A `bool`, if 'True' the output tensor is sorted
      according to the sorted batch ids of `point_cloud`.
    return_padded: A `bool`, if 'True' the output tensor is sorted and
      zero padded.

  Returns:
    A `float` `Tensor` of shape
      `[N, C]`, if `return_padded` is `False`
    or
      `[A1, ..., An, V, C]`, if `return_padded` is `True`.

  """

  if return_padded:
    unflatten = point_cloud.get_unflatten()
    features = unflatten(features)
  elif return_sorted:
    features = tf.gather(features, point_cloud._sorted_indices_batch)
  return features


def random_rotation(points, name=None):
  """ Method to rotate 3D points randomly.

  Args:
    points: A `float` `Tensor` of shape `[N, 3]`.

  Returns:
    A `float` `Tensor` of the same shape as `points`.

  """
  with tf.compat.v1.name_scope(name, 'random point rotation', [points]):
    points = tf.convert_to_tensor(value=points)
    angles = tf.random.uniform([3], 0, 2 * np.pi)
    sine = tf.math.sin(angles)
    cosine = tf.math.cos(angles)
    Rx = tf.stack(([1.0, 0.0, 0.0],
                   [0.0, cosine[0], -sine[0]],
                   [0.0, sine[0], cosine[0]]), axis=1)
    Ry = tf.stack(([cosine[1], 0, sine[1]],
                   [0.0, 1.0, 0.0],
                   [-sine[1], 0.0, cosine[1]]), axis=1)
    Rz = tf.stack(([cosine[2], -sine[2], 0.0],
                   [sine[2], cosine[2], 0.0],
                   [0.0, 0.0, 1.0]), axis=1)
    R = tf.matmul(tf.matmul(Rx, Ry), Rz)
    return tf.matmul(points, R)


def _hexagon(scale, z_shift):
  """ Numpy hexagon ponts in the xy-plane with diameter `scale`
  at z-position `z_shift`.

  Args:
    scale: A `float`.
    z_shift: A `float`.

  Returns:
    A `np.array` of shape `[6, 3]`.

  """
  phi = np.sqrt(3) / 2
  points = [[0, 1, 0],
            [0, -1, 0],
            [0.5, phi, 0],
            [0.5, -phi, 0],
            [-0.5, phi, 0],
            [0.5, -phi, 0]]
  points = np.array(points) * scale
  points[:, 2] += z_shift
  return points


def _pentagon(scale, z_shift):
  """ Numpy pentagon ponts in the xy-plane with diameter `scale`
  at z-position `z_shift`.

  Args:
    scale: A `float`.
    z_shift: A `float`.

  Returns:
    A `np.array` of shape `[5, 3]`.

  """
  c1 = (np.sqrt(5) - 1) / 4
  c2 = (np.sqrt(5) + 1) / 4
  s1 = np.sqrt(10 + 2 * np.sqrt(5)) / 4
  s2 = np.sqrt(10 - 2 * np.sqrt(5)) / 4
  points = [[1, 0, 0],
            [c1, s1, 0],
            [c1, -s1, 0],
            [-c2, s2, 0],
            [-c2, -s2, 0]]
  points = np.array(points) * scale
  points[:, 2] += z_shift
  return points


def square(scale, z_shift):
  """ Numpy square ponts in the xy-plane with diameter `scale`
  at z-position `z_shift`.

  Args:
    scale: A `float`.
    z_shift: A `float`.

  Returns:
    A `np.array` of shape `[4, 3]`.

  """
  points = [[1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0]]
  points = np.array(points) * scale
  points[:, 2] += z_shift
  return points


def spherical_kernel_points(num_points, rotate=True, name=None):
  """ Kernel points in a unit sphere.

  The points are located at positions as described in Appendix B of
  [KPConv: Flexible and Deformable Convolution for Point Clouds. Thomas et
  al.,
  2019](https://arxiv.org/abs/1904.08889).

  Args:
    num_points: An `int`, the number of kernel points, must be in
      `[5, 7, 13, 15, 18]`.
    rotate: A 'bool', if `True` a random rotation is applied to the points.

  Returns:
    A `float` `Tensor` of shape `[num_points, 3]`.

  """
  with tf.compat.v1.name_scope(
      name, "KPConv kernel points",
      [num_points, rotate]):

    if num_points not in [5, 7, 13, 15, 18]:
      raise ValueError('KPConv currently only supports kernel sizes' + \
                       ' [5, 7, 13, 15, 18]')
    if num_points == 5:
      # Tetrahedron
      points = tf.Variable([[0, 0, 0],
                            [0, 0, 1],
                            [tf.sqrt(8 / 9), 0, -1 / 3],
                            [- tf.sqrt(2 / 9), tf.sqrt(2 / 3), - 1 / 3],
                            [-tf.sqrt(2 / 9), - tf.sqrt(2 / 3), -1 / 3]],
                           dtype=tf.float32)
    elif num_points == 7:
      # Octahedron
      points = tf.Variable([[0, 0, 0],
                            [1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 0, -1]], dtype=tf.float32)
    elif num_points == 13:
      # Icosahedron
      phi = (1 + tf.sqrt(5)) / 2
      points = tf.Variable([[0, 0, 0],
                            [0, 1, phi],
                            [0, 1, -phi],
                            [0, -1, phi],
                            [0, -1, -phi],
                            [1, phi, 0],
                            [1, -phi, 0],
                            [-1, phi, 0],
                            [-1, -phi, 0],
                            [phi, 0, 1],
                            [-phi, 0, 1],
                            [phi, 0, -1],
                            [-phi, 0, -1]], dtype=tf.float32)
    elif num_points == 15:
      hex1 = _hexagon(0.5, np.sqrt(3) / 2)
      hex2 = _hexagon(-0.5, np.sqrt(3) / 2)[:,[1, 0, 2]]  # rotated 90 deg in xy
      points = np.concatenate(([[0, 0, 0], [0, 0, 1], [0, 0, -1]], hex1, hex2),
                              axis=0)
      points = tf.Variable(points, dtype=tf.float32)
    elif num_points == 18:
      penta1 = _pentagon(1 / np.sqrt(2), 0.5)
      penta2 = -_pentagon(1.0, 0.0) # flipped in xy
      penta3 = _pentagon(-1 / np.sqrt(2), 0.5)
      points = np.concatenate(([[0, 0, 0], [0, 0, 1], [0, 0, -1]],
                               penta1, penta2, penta3),
                              axis=0)
      points = tf.Variable(points, dtype=tf.float32)
    if rotate:
      points = random_rotation(points)
    return points


def cube_kernel_points(cbrt_num_points, name):
  """ Regularily distributed points in a unit cube.

  Args:
    cbrt_num_points: An `int`, the cubic root of the number of points.

  Returns:
    A `float` `Tensor` of shape `[cbrt_num_points^3, 3]`.

  """

  with tf.compat.v1.name_scope(name, "box kernel points", [cbrt_num_points]):
    x = np.linspace(0, 1, cbrt_num_points)
    x, y, z = np.meshgrid(x, x, x)
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    return tf.Variable(points, dtype=tf.float32)


def _identity(features, *args, **kwargs):
  """ Simple identity layer, to be used as placeholder.

  Used to replace projection shortcuts, if not desired.

  """
  return features
