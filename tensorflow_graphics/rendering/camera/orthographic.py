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
r"""This module implements orthographic camera functionalities.

An orthographic camera represents three-dimensional objects in two dimensions
by parallel projection, in which the projection lines are parallel to the
camera axis. The camera axis is the line perpendicular to the image plane
starting at the camera center.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def project(point_3d, name="orthographic_project"):
  r"""Projects a 3d point onto the 2d camera plane.

  Projects a 3d point \\((x, y, z)\\) to a 2d point \\((x', y')\\) onto the
  image plane, with

  $$
  \begin{matrix}
  x' = x, & y' = y.
  \end{matrix}
  $$

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point_3d: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to project.
    name: A name for this op that defaults to "orthographic_project".

  Returns:
    A tensor of shape `[A1, ..., An, 2]`, where the last dimension represents
    a 2d point.

  Raises:
    ValueError: If the shape of `point_3d` is not supported.
  """
  with tf.name_scope(name):
    point_3d = tf.convert_to_tensor(value=point_3d)

    shape.check_static(
        tensor=point_3d, tensor_name="point_3d", has_dim_equals=(-1, 3))

    point_xy, _ = tf.split(point_3d, (2, 1), axis=-1)
    return point_xy


def ray(point_2d, name="orthographic_ray"):
  r"""Computes the 3d ray for a 2d point (the z component of the ray is 1).

  Computes the 3d ray \\((r_x, r_y, 1)\\) for a 2d point \\((x', y')\\) on the
  image plane. For an orthographic camera the rays are constant over the image
  plane with

  $$
  \begin{matrix}
  r_x = 0, & r_y = 0, & z = 1.
  \end{matrix}
  $$

  Note: In the following, A1 to An are optional batch dimensions.

  Args:
    point_2d: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point.
    name: A name for this op that defaults to "orthographic_ray".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d ray.

  Raises:
    ValueError: If the shape of `point_2d` is not supported.
  """
  with tf.name_scope(name):
    point_2d = tf.convert_to_tensor(value=point_2d)

    shape.check_static(
        tensor=point_2d, tensor_name="point_2d", has_dim_equals=(-1, 2))

    ones = tf.ones_like(point_2d[..., :1])
    # point_2d is multiplied by zero to ensure it has defined gradients.
    return tf.concat((point_2d * 0.0, ones), axis=-1)


def unproject(point_2d, depth, name="orthographic_unproject"):
  r"""Unprojects a 2d point in 3d.

  Unprojects a 2d point \\((x', y')\\) to a 3d point \\((x, y, z)\\) given its
  depth \\(z\\), with

  $$
  \begin{matrix}
  x = x', & y = y', & z = z.
  \end{matrix}
  $$

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point_2d: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
      represents a 2d point to unproject.
    depth: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents the depth of a 2d point.
    name: A name for this op that defaults to "orthographic_unproject".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.

  Raises:
    ValueError: If the shape of `point_2d`, `depth` is not supported.
  """
  with tf.name_scope(name):
    point_2d = tf.convert_to_tensor(value=point_2d)
    depth = tf.convert_to_tensor(value=depth)

    shape.check_static(
        tensor=point_2d, tensor_name="point_2d", has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=depth, tensor_name="depth", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(point_2d, depth),
        tensor_names=("point_2d", "depth"),
        last_axes=-2,
        broadcast_compatible=False)

    return tf.concat((point_2d, depth), axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
