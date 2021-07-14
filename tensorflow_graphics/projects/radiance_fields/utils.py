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
"""Functions for helping running NeRF-based networks."""
import math
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.util import shape

pi = math.pi


def match_intermediate_batch_dimensions(tensor1, tensor2):
  """Match the batch dimensions.

  Args:
    tensor1: A tensor of shape `[A1, M]`.
    tensor2: A tensor of shape `[A1, ..., An, N]`.
  Returns:
    A tensor of shape `[A1, ..., An, M]`.
  """
  shape.check_static(
      tensor=tensor1,
      tensor_name="tensor1",
      has_rank=2)
  shape.check_static(
      tensor=tensor2,
      tensor_name="tensor2",
      has_rank_greater_than=1)
  shape.compare_dimensions(tensors=(tensor1, tensor2),
                           tensor_names=("tensor1", "tensor2"),
                           axes=0)

  shape1 = tf.shape(tensor1)
  shape2 = tf.shape(tensor2)
  shape_diff = len(shape2) - len(shape1)
  new_shape = tf.concat([[shape1[0]], [1]*shape_diff, [shape1[-1]]], axis=-1)
  target_shape = tf.concat([shape2[:-1], [shape1[-1]]], axis=-1)
  return tf.broadcast_to(tf.reshape(tensor1, new_shape), target_shape)


def change_coordinate_system(points3d,
                             rotations=(0., 0., 0.),
                             scale=(1., 1., 1.),
                             name="change_coordinate_system"):
  """Change coordinate system.

  Args:
      points3d: A tensor of shape `[A1, ..., An, M, 3]` containing the
        3D position of M points.
      rotations: A tuple containing the X, Y, Z axis rotation.
      scale: A tuple containing the X, Y, Z axis scale.
      name: A name for this op. Defaults to "change_coordinate_system".

  Returns:
      [type]: [description]
  """
  with tf.name_scope(name):
    points3d = tf.convert_to_tensor(points3d)
    rotation = tf.convert_to_tensor(rotations)
    scale = tf.convert_to_tensor(scale)

    rotation_matrix = rotation_matrix_3d.from_euler(rotation)
    scaling_matrix = scale*tf.eye(3, 3)

    target_shape = [1]*(len(points3d.get_shape().as_list())- 2) + [3, 3]
    transformation = tf.matmul(scaling_matrix, rotation_matrix)
    transformation = tf.reshape(transformation, target_shape)
    return tf.linalg.matrix_transpose(
        tf.matmul(transformation, tf.linalg.matrix_transpose(points3d)))


@tf.function
def get_distances_between_points(ray_points3d, last_bin_witdh=1e10):
  """Estimates the distance between points in a ray.

  Args:
    ray_points3d: A tensor of shape `[A1, ..., An, M, 3]`,
      where M is the number of points in a ray.
    last_bin_witdh: A scalar indicating the witdth of the last bin.

  Returns:
    A tensor of shape `[A1, ..., An, M]` containing the distances between
      the M points, with the distance of the last element set to a high value.
  """
  shape.check_static(
      tensor=ray_points3d,
      tensor_name="ray_points3d",
      has_dim_equals=(-1, 3))
  shape.check_static(
      tensor=ray_points3d,
      tensor_name="ray_points3d",
      has_rank_greater_than=1)
  dists = tf.norm(ray_points3d[..., 1:, :] - ray_points3d[..., :-1, :], axis=-1)
  return tf.concat([dists, tf.broadcast_to([last_bin_witdh],
                                           dists[..., :1].shape)], axis=-1)


@tf.function
def _move_in_front_of_camera(points3d,
                             rotation_matrix,
                             translation_vector):
  """Moves a set of points in front of a camera given by its extrinsics.

  Args:
    points3d: A tensor of shape `[A1, ..., An, M, 3]`,
      where M is the number of points.
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`.
    translation_vector: A tensor of shape `[A1, ..., An, 3, 1]`.
  Returns:
    A tensor of shape `[A1, ..., An, M, 3]`.
  """
  points3d = tf.convert_to_tensor(value=points3d)
  rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)
  translation_vector = tf.convert_to_tensor(value=translation_vector)

  shape.check_static(
      tensor=points3d, tensor_name="points3d", has_dim_equals=(-1, 3))
  shape.check_static(
      tensor=rotation_matrix,
      tensor_name="rotation_matrix",
      has_dim_equals=(-1, 3))
  shape.check_static(
      tensor=rotation_matrix,
      tensor_name="rotation_matrix",
      has_dim_equals=(-2, 3))
  shape.check_static(
      tensor=translation_vector,
      tensor_name="translation_vector",
      has_dim_equals=(-1, 1))
  shape.check_static(
      tensor=translation_vector,
      tensor_name="translation_vector",
      has_dim_equals=(-2, 3))
  shape.compare_batch_dimensions(
      tensors=(points3d, rotation_matrix, translation_vector),
      tensor_names=("points3d", "rotation_matrix", "translation_vector"),
      last_axes=-3,
      broadcast_compatible=True)

  points3d_corrected = tf.linalg.matrix_transpose(points3d) + translation_vector
  rotation_matrix_t = -tf.linalg.matrix_transpose(rotation_matrix)
  points3d_world = tf.matmul(rotation_matrix_t, points3d_corrected)
  return tf.linalg.matrix_transpose(points3d_world)


@tf.function
def camera_rays_from_extrinsics(rays,
                                rotation_matrix,
                                translation_vector):
  """Transform the rays from a camera located at (0, 0, 0) to ray origins and directions for a camera with given extrinsics.

  Args:
    rays: A tensor of shape `[A1, ..., An, N, 3]` where N is the number of rays.
    rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`.
    translation_vector: A tensor of shape `[A1, ..., An, 3, 1]`.
  Returns:
    A tensor of shape `[A1, ..., An, N, 3]` representing the ray origin and
    a tensor of shape `[A1, ..., An, N, 3]` representing the ray direction.
  """
  shape.check_static(tensor=rays,
                     tensor_name="pixels",
                     has_rank_greater_than=1)
  shape.compare_batch_dimensions(
      tensors=(rays, rotation_matrix, translation_vector),
      tensor_names=("points_on_rays", "rotation_matrix", "translation_vector"),
      last_axes=-3,
      broadcast_compatible=False)

  rays_org = _move_in_front_of_camera(tf.zeros_like(rays),
                                      rotation_matrix,
                                      translation_vector)
  rays_dir_ = _move_in_front_of_camera(rays,
                                       rotation_matrix,
                                       0 * translation_vector)
  rays_dir = rays_dir_/tf.norm(rays_dir_, axis=-1, keepdims=True)
  return rays_org, rays_dir


def camera_rays_from_transformation_matrix(rays, transform_matrix):
  """Estimate ray origin and direction from transformation matrix.

  Args:
    rays: A tensor of shape `[A1, ..., An, N, 3]` where N is the number of rays.
    transform_matrix: A tensor of shape `[A1, ..., An, 4, 4]`.
  Returns:
    A tensor of shape `[A1, ..., An, N, 3]` representing the ray origin and
    a tensor of shape `[A1, ..., An, N, 3]` representing the ray direction.
  """
  rays_o = transform_matrix[..., :3, -1]  # [A1, ..., An, 3]
  rays_o = tf.expand_dims(rays_o, -2)  # [A1, ..., An, 1, 3]
  rays_o = tf.broadcast_to(rays_o, tf.shape(rays))  # [A1, ..., An, N, 3]
  rot = transform_matrix[..., tf.newaxis, :3, :3]
  rays_d = tf.reduce_sum(tf.expand_dims(rays, axis=-2) * rot, axis=-1)
  return rays_o, rays_d


def l2_loss(prediction, target, weights=1.0):
  """L2 loss implementation that forces same prediction and target shapes."""
  assert prediction.shape == target.shape, "Shape dims should be the same."
  return tf.reduce_mean(weights * tf.square(target - prediction))
