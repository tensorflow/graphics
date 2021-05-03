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
"""Tensorflow ray utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.math import sampling
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util.type_alias import TensorLike


def _points_from_z_values(ray_org: TensorLike,
                          ray_dir: TensorLike,
                          z_values: TensorLike) -> tf.Tensor:
  """Sample points on rays given the z values (distances along the rays).

  Args:
    ray_org: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D position of the ray origin.
    ray_dir: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D direction of the ray.
    z_values: A tensor of shape `[A1, ..., An, M]` containing the 1D position of
      M points along the ray.

  Returns:
    A tensor of shape `[A1, ..., An, M, 3]`
  """
  shape.check_static(
      tensor=ray_dir,
      tensor_name="ray_dir",
      has_dim_equals=(-1, 3))
  shape.check_static(
      tensor=ray_org,
      tensor_name="ray_org",
      has_dim_equals=(-1, 3))
  shape.compare_batch_dimensions(
      tensors=(ray_org, ray_dir, z_values),
      tensor_names=("ray_org", "ray_dir", "z_values"),
      last_axes=-2,
      broadcast_compatible=False)

  points3d = (tf.expand_dims(ray_dir, axis=-2) *
              tf.expand_dims(z_values, axis=-1))
  points3d = tf.expand_dims(ray_org, -2) + points3d
  return points3d


def sample_stratified_1d(
    ray_org: TensorLike,
    ray_dir: TensorLike,
    near: float,
    far: float,
    n_samples: int,
    name: str = "sample_stratified_1d") -> Tuple[tf.Tensor, tf.Tensor]:
  """Sample points on a ray using stratified sampling.

  Args:
    ray_org: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D position of the ray origin.
    ray_dir: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D direction of the ray.
    near: The smallest distance from the ray origin that a sample can have.
    far: The largest distance from the ray origin that a sample can have.
    n_samples: A number M to sample on the ray.
    name: A name for this op that defaults to "stratified_sampling".

  Returns:
    A tensor of shape `[A1, ..., An, M, 3]` indicating the M points on the ray
      and a tensor of shape `[A1, ..., An, M]` for the Z values on the points.
  """
  with tf.name_scope(name):
    shape.check_static(
        tensor=ray_org,
        tensor_name="ray_org",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=ray_dir,
        tensor_name="ray_dir",
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(ray_org, ray_dir),
        tensor_names=("ray_org", "ray_dir"),
        last_axes=(-2, -2),
        broadcast_compatible=False)

    batch_dims = tf.shape(ray_org)[:-1]
    random_z_values = sampling.stratified_1d(near * tf.ones(batch_dims),
                                             far * tf.ones(batch_dims),
                                             n_samples)
    points3d = _points_from_z_values(ray_org, ray_dir, random_z_values)
    return points3d, random_z_values


def sample_inverse_transform_stratified_1d(
    ray_org: TensorLike,
    ray_dir: TensorLike,
    z_values_init: TensorLike,
    weights_init: TensorLike,
    n_samples: int,
    combine_z_values=True,
    name: str = "sample_inverse_transform_stratified_1d"):
  """Sample points on a ray using inverse transform stratified sampling.

  The rays are defined by their origin and direction. Along each ray, there are
  M samples (provided as 1D distances from the ray origin) and the corresponding
  weights (probabilities) that facilitate the inverse transform sampling.

  Args:
    ray_org: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D position of the ray origin.
    ray_dir: A tensor of shape `[A1, ..., An, 3]`,
      where the last dimension represents the 3D direction of the ray.
    z_values_init: A tensor of shape `[A1, ..., An, M]`,
      where the last dimension is the location of M points along the ray.
    weights_init: A tensor of shape `[A1, ..., An, M]`,
      where the last dimension is the density of M points along the ray.
    n_samples: A number M to sample on the ray.
    combine_z_values: Wether to combine the new 1D samples with
      the initial points.
    name: A name for this op that defaults to "stratified_sampling".

  Returns:
    A tensor of shape `[A1, ..., An, M, 3]` indicating the M points on the ray
      and a tensor of shape `[A1, ..., An, M]` for the Z values on the points.
  """
  with tf.name_scope(name):
    shape.check_static(
        tensor=ray_org,
        tensor_name="ray_org",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=ray_dir,
        tensor_name="ray_dir",
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(ray_org, ray_dir, z_values_init, weights_init),
        tensor_names=("ray_org", "ray_dir", "z_values_init", "weights_init"),
        last_axes=-2,
        broadcast_compatible=False)
    shape.compare_dimensions(
        tensors=(z_values_init, weights_init),
        tensor_names=("z_values_init", "weights_init"),
        axes=-1)

    bin_start = z_values_init[..., :-1]
    bin_width = z_values_init[..., 1:] - z_values_init[..., :-1]
    bin_weights = .5 * (weights_init[..., 1:] + weights_init[..., :-1])
    random_z_values = sampling.inverse_transform_stratified_1d(bin_start,
                                                               bin_width,
                                                               bin_weights,
                                                               n_samples)
    random_z_values = tf.stop_gradient(random_z_values)
    if combine_z_values:
      z_values_final = tf.sort(tf.concat([z_values_init,
                                          random_z_values], -1), -1)
    else:
      z_values_final = tf.sort(random_z_values, -1)
    points3d = _points_from_z_values(ray_org, ray_dir, z_values_final)
    return points3d, z_values_final


def triangulate(startpoints, endpoints, weights, name="ray_triangulate"):
  """Triangulates 3d points by miminizing the sum of squared distances to rays.

  The rays are defined by their start points and endpoints. At least two rays
  are required to triangulate any given point. Contrary to the standard
  reprojection-error metric, the sum of squared distances to rays can be
  minimized in a closed form.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    startpoints: A tensor of ray start points with shape `[A1, ..., An, V, 3]`,
      the number of rays V around which the solution points live should be
      greater or equal to 2, otherwise triangulation is impossible.
    endpoints: A tensor of ray endpoints with shape `[A1, ..., An, V, 3]`, the
      number of rays V around which the solution points live should be greater
      or equal to 2, otherwise triangulation is impossible. The `endpoints`
      tensor should have the same shape as the `startpoints` tensor.
    weights: A tensor of ray weights (certainties) with shape `[A1, ..., An,
      V]`. Weights should have all positive entries. Weight should have at least
      two non-zero entries for each point (at least two rays should have
      certainties > 0).
    name: A name for this op. The default value of None means "ray_triangulate".

  Returns:
    A tensor of triangulated points with shape `[A1, ..., An, 3]`.

  Raises:
    ValueError: If the shape of the arguments is not supported.
  """
  with tf.name_scope(name):
    startpoints = tf.convert_to_tensor(value=startpoints)
    endpoints = tf.convert_to_tensor(value=endpoints)
    weights = tf.convert_to_tensor(value=weights)

    shape.check_static(
        tensor=startpoints,
        tensor_name="startpoints",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3),
        has_dim_greater_than=(-2, 1))
    shape.check_static(
        tensor=endpoints,
        tensor_name="endpoints",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3),
        has_dim_greater_than=(-2, 1))
    shape.compare_batch_dimensions(
        tensors=(startpoints, endpoints, weights),
        last_axes=(-2, -2, -1),
        broadcast_compatible=False)
    weights = asserts.assert_all_above(weights, 0.0, open_bound=False)
    weights = asserts.assert_at_least_k_non_zero_entries(weights, k=2)

    left_hand_side_list = []
    right_hand_side_list = []
    # TODO(b/130892100): Replace the inefficient for loop and add comments here.
    for ray_id in range(weights.shape[-1]):
      weights_single_ray = weights[..., ray_id]
      startpoints_single_ray = startpoints[..., ray_id, :]
      endpoints_singleview = endpoints[..., ray_id, :]
      ray = endpoints_singleview - startpoints_single_ray
      ray = tf.nn.l2_normalize(ray, axis=-1)
      ray_x, ray_y, ray_z = tf.unstack(ray, axis=-1)
      zeros = tf.zeros_like(ray_x)
      cross_product_matrix = tf.stack(
          (zeros, -ray_z, ray_y, ray_z, zeros, -ray_x, -ray_y, ray_x, zeros),
          axis=-1)
      cross_product_matrix_shape = tf.concat(
          (tf.shape(input=cross_product_matrix)[:-1], (3, 3)), axis=-1)
      cross_product_matrix = tf.reshape(
          cross_product_matrix, shape=cross_product_matrix_shape)
      weights_single_ray = tf.expand_dims(weights_single_ray, axis=-1)
      weights_single_ray = tf.expand_dims(weights_single_ray, axis=-1)
      left_hand_side = weights_single_ray * cross_product_matrix
      left_hand_side_list.append(left_hand_side)
      dot_product = tf.matmul(cross_product_matrix,
                              tf.expand_dims(startpoints_single_ray, axis=-1))
      right_hand_side = weights_single_ray * dot_product
      right_hand_side_list.append(right_hand_side)
    left_hand_side_multi_rays = tf.concat(left_hand_side_list, axis=-2)
    right_hand_side_multi_rays = tf.concat(right_hand_side_list, axis=-2)
    points = tf.linalg.lstsq(left_hand_side_multi_rays,
                             right_hand_side_multi_rays)
    points = tf.squeeze(points, axis=-1)

    return points


# TODO(b/130893491): Add batch support for radii and return [A1, ... , 3, 2].
def intersection_ray_sphere(sphere_center,
                            sphere_radius,
                            ray,
                            point_on_ray,
                            name="ray_intersection_ray_sphere"):
  """Finds positions and surface normals where the sphere and the ray intersect.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sphere_center: A tensor of shape `[3]` representing the 3d sphere center.
    sphere_radius: A tensor of shape `[1]` containing a strictly positive value
      defining the radius of the sphere.
    ray: A tensor of shape `[A1, ..., An, 3]` containing normalized 3D vectors.
    point_on_ray: A tensor of shape `[A1, ..., An, 3]`.
    name: A name for this op. The default value of None means
      "ray_intersection_ray_sphere".

  Returns:
    A tensor of shape `[2, A1, ..., An, 3]` containing the position of the
    intersections, and a tensor of shape `[2, A1, ..., An, 3]` the associated
    surface normals at that point. Both tensors contain NaNs when there is no
    intersections. The first dimension of the returned tensor provides access to
    the first and second intersections of the ray with the sphere.

  Raises:
    ValueError: if the shape of `sphere_center`, `sphere_radius`, `ray` or
      `point_on_ray` is not supported.
    tf.errors.InvalidArgumentError: If `ray` is not normalized.
  """
  with tf.name_scope(name):
    sphere_center = tf.convert_to_tensor(value=sphere_center)
    sphere_radius = tf.convert_to_tensor(value=sphere_radius)
    ray = tf.convert_to_tensor(value=ray)
    point_on_ray = tf.convert_to_tensor(value=point_on_ray)

    shape.check_static(
        tensor=sphere_center,
        tensor_name="sphere_center",
        has_rank=1,
        has_dim_equals=(0, 3))
    shape.check_static(
        tensor=sphere_radius,
        tensor_name="sphere_radius",
        has_rank=1,
        has_dim_equals=(0, 1))
    shape.check_static(tensor=ray, tensor_name="ray", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=point_on_ray, tensor_name="point_on_ray", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(ray, point_on_ray),
        last_axes=(-2, -2),
        broadcast_compatible=False)
    sphere_radius = asserts.assert_all_above(
        sphere_radius, 0.0, open_bound=True)
    ray = asserts.assert_normalized(ray)

    vector_sphere_center_to_point_on_ray = sphere_center - point_on_ray
    distance_sphere_center_to_point_on_ray = tf.norm(
        tensor=vector_sphere_center_to_point_on_ray, axis=-1, keepdims=True)
    distance_projection_sphere_center_on_ray = vector.dot(
        vector_sphere_center_to_point_on_ray, ray)
    closest_distance_sphere_center_to_ray = tf.sqrt(
        tf.square(distance_sphere_center_to_point_on_ray) -
        tf.pow(distance_projection_sphere_center_on_ray, 2))
    half_secant_length = tf.sqrt(
        tf.square(sphere_radius) -
        tf.square(closest_distance_sphere_center_to_ray))
    distances = tf.stack(
        (distance_projection_sphere_center_on_ray - half_secant_length,
         distance_projection_sphere_center_on_ray + half_secant_length),
        axis=0)
    intersections_points = distances * ray + point_on_ray
    normals = tf.math.l2_normalize(
        intersections_points - sphere_center, axis=-1)
    return intersections_points, normals


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
