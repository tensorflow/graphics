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
# Lint as: python3
"""Computes a weighted point sampling of a triangular mesh.

This op computes a uniform sampling of points on the surface of the mesh.
Points are sampled from the surface of each triangle using a uniform
distribution, proportional to a specified face density (e.g. face area).

Uses the approach mentioned in the TOG 2002 paper "Shape distributions"
(https://dl.acm.org/citation.cfm?id=571648)
to generate random barycentric coordinates.

This op can be used for several tasks, including better mesh reconstruction.
For example, see these recent papers demonstrating reconstruction losses using
this op:
1. "GEOMetrics: Exploiting Geometric Structure for Graph-Encoded Objects"
(https://arxiv.org/abs/1901.11461) ICML 2019.
2. "Mesh R-CNN" (https://arxiv.org/abs/1906.02739) ICCV 2019.

Op is differentiable w.r.t mesh vertex positions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Tuple

import tensorflow as tf

from tensorflow_graphics.geometry.representation import triangle
from tensorflow_graphics.geometry.representation.mesh import normals
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def triangle_area(vertex0: type_alias.TensorLike,
                  vertex1: type_alias.TensorLike,
                  vertex2: type_alias.TensorLike,
                  name: str = "triangle_area") -> type_alias.TensorLike:
  """Computes triangle areas.

  Note:
    Computed triangle area = 0.5 * | e1 x e2 | where e1 and e2 are edges
      of triangle.

    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

    In the following, A1 to An are optional batch dimensions.

  Args:
    vertex0: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vertex of a triangle.
    vertex1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vertex of a triangle.
    vertex2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the third vertex of a triangle.
    name: A name for this op. Defaults to "triangle_area".

  Returns:
    A tensor of shape `[A1, ..., An, 1]`, where the last dimension represents
      the triangle areas.
  """
  with tf.name_scope(name):
    vertex0 = tf.convert_to_tensor(value=vertex0)
    vertex1 = tf.convert_to_tensor(value=vertex1)
    vertex2 = tf.convert_to_tensor(value=vertex2)

    triangle_normals = triangle.normal(
        vertex0, vertex1, vertex2, normalize=False)
    areas = 0.5 * tf.linalg.norm(tensor=triangle_normals, axis=-1)
    return areas


def _random_categorical_sample(
    num_samples: int,
    weights: type_alias.TensorLike,
    seed: Optional[type_alias.TensorLike] = None,
    stateless: bool = False,
    name: str = "random_categorical_sample",
    sample_dtype: tf.DType = tf.int32) -> type_alias.TensorLike:
  """Samples from a categorical distribution with arbitrary batch dimensions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    num_samples: An `int32` scalar denoting the number of samples to generate
      per mesh.
    weights: A `float` tensor of shape `[A1, ..., An, F]` where F is number of
      faces.
      All weights must be > 0.
    seed: Optional random seed, value depends on `stateless`.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then `seed` must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate the same reproducible sequence
      across calls. If stateless=False, then a stateful random number generator
      is used (default behavior).
    name: Name for op. Defaults to "random_categorical_sample".
    sample_dtype: Type of output samples.

  Returns:
    A `sample_dtype` tensor of shape `[A1, ..., An, num_samples]`.
  """
  with tf.name_scope(name):
    asserts.assert_all_above(weights, 0)
    logits = tf.math.log(weights)
    num_faces = tf.shape(input=logits)[-1]
    batch_shape = tf.shape(input=logits)[:-1]
    logits_2d = tf.reshape(logits, [-1, num_faces])
    if stateless:
      seed = tf.convert_to_tensor(value=seed)
      shape.check_static(
          tensor=seed, tensor_name="seed", has_dim_equals=(-1, 2))
      sample_fn = tf.random.stateless_categorical
    else:
      sample_fn = tf.random.categorical
    draws = sample_fn(
        logits=logits_2d,
        num_samples=num_samples,
        dtype=sample_dtype,
        seed=seed)
    samples = tf.reshape(
        draws,
        shape=tf.concat((batch_shape, (num_samples,)), axis=0))
    return samples


def generate_random_face_indices(
    num_samples: int,
    face_weights: type_alias.TensorLike,
    seed: Optional[type_alias.TensorLike] = None,
    stateless: bool = False,
    name: str = "generate_random_face_indices") -> type_alias.TensorLike:
  """Generate a sample of face ids given per face probability.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    num_samples: An `int32` scalar denoting the number of samples to generate
      per mesh.
    face_weights: A `float` tensor of shape `[A1, ..., An, F]` where F is
      number of faces. All weights must be > 0.
    seed: Optional seed for the random number generator.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then `seed` must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate the same reproducible sequence
      across calls. If stateless=False, then a stateful random number generator
      is used (default behavior).
    name: Name for op. Defaults to "generate_random_face_indices".

  Returns:
    An `int32` tensor of shape `[A1, ..., An, num_samples]` denoting sampled
      face indices.
  """
  with tf.name_scope(name):
    num_samples = tf.convert_to_tensor(value=num_samples)
    face_weights = tf.convert_to_tensor(value=face_weights)
    shape.check_static(
        tensor=face_weights,
        tensor_name="face_weights",
        has_rank_greater_than=0)
    shape.check_static(
        tensor=num_samples, tensor_name="num_samples", has_rank=0)

    face_weights = asserts.assert_all_above(face_weights, minval=0.0)
    eps = asserts.select_eps_for_division(face_weights.dtype)
    face_weights = face_weights + eps
    sampled_face_indices = _random_categorical_sample(
        num_samples=num_samples,
        weights=face_weights,
        seed=seed,
        stateless=stateless)
    return sampled_face_indices


def generate_random_barycentric_coordinates(
    sample_shape: type_alias.TensorLike,
    dtype: tf.DType = tf.dtypes.float32,
    seed: Optional[type_alias.TensorLike] = None,
    stateless: bool = False,
    name: str = "generate_random_barycentric_coordinates"
) -> type_alias.TensorLike:
  """Generate uniformly sampled random barycentric coordinates.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    sample_shape: An `int` tensor with shape `[n+1,]` and values `(A1, ..., An,
      num_samples)` denoting total number of random samples drawn, where `n` is
      number of batch dimensions, and `num_samples` is the number of samples
      drawn for each mesh.
    dtype: Optional type of generated barycentric coordinates, defaults to
      float32.
    seed: An optional random seed.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then `seed` must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate the same reproducible sequence
      across calls. If stateless=False, then a stateful random number generator
      is used (default behavior).
    name: Name for op. Defaults to "generate_random_barycentric_coordinates".

  Returns:
    A `dtype` tensor of shape [A1, ..., An, num_samples, 3],
      where the last dimension contains the sampled barycentric coordinates.


  """
  with tf.name_scope(name):
    sample_shape = tf.convert_to_tensor(value=sample_shape)
    shape.check_static(
        tensor=sample_shape, tensor_name="sample_shape", has_rank=1)
    sample_shape = tf.concat((sample_shape, (2,)), axis=0)

    if stateless:
      seed = tf.convert_to_tensor(value=seed)
      shape.check_static(
          tensor=seed, tensor_name="seed", has_dim_equals=(-1, 2))
      sample_fn = tf.random.stateless_uniform
    else:
      sample_fn = tf.random.uniform
    random_uniform = sample_fn(
        shape=sample_shape, minval=0.0, maxval=1.0, dtype=dtype, seed=seed)
    random1 = tf.sqrt(random_uniform[..., 0])
    random2 = random_uniform[..., 1]
    barycentric = tf.stack(
        (1 - random1, random1 * (1 - random2), random1 * random2), axis=-1)
    return barycentric


def weighted_random_sample_triangle_mesh(
    vertex_attributes: type_alias.TensorLike,
    faces: type_alias.TensorLike,
    num_samples: int,
    face_weights: type_alias.TensorLike,
    seed: Optional[type_alias.TensorLike] = None,
    stateless: bool = False,
    name: str = "weighted_random_sample_triangle_mesh"
) -> Tuple[type_alias.TensorLike, type_alias.TensorLike]:
  """Performs a face probability weighted random sampling of a tri mesh.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertex_attributes: A `float` tensor of shape `[A1, ..., An, V, D]`, where V
      is the number of vertices, and D is dimensionality of each vertex.
    faces: A `int` tensor of shape `[A1, ..., An, F, 3]`, where F is the number
      of faces.
    num_samples: A `int` 0-D tensor denoting number of samples to be drawn from
      each mesh.
    face_weights: A `float` tensor of shape ``[A1, ..., An, F]`, denoting
      unnormalized sampling probability of each face, where F is the number of
      faces.
    seed: Optional random seed.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then seed must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate same sequence across calls.
    name: Name for op. Defaults to "weighted_random_sample_triangle_mesh".

  Returns:
    sample_points: A `float` tensor of shape `[A1, ..., An, num_samples, D]`,
      where D is dimensionality of each sampled point.
    sample_face_indices: A `int` tensor of shape `[A1, ..., An, num_samples]`.
  """
  with tf.name_scope(name):
    faces = tf.convert_to_tensor(value=faces)
    vertex_attributes = tf.convert_to_tensor(value=vertex_attributes)
    face_weights = tf.convert_to_tensor(value=face_weights)
    num_samples = tf.convert_to_tensor(value=num_samples)

    shape.check_static(
        tensor=vertex_attributes,
        tensor_name="vertex_attributes",
        has_rank_greater_than=1)
    shape.check_static(
        tensor=faces, tensor_name="faces", has_rank_greater_than=1)
    shape.check_static(
        tensor=face_weights,
        tensor_name="face_weights",
        has_rank_greater_than=0)
    shape.compare_batch_dimensions(
        tensors=(faces, face_weights),
        last_axes=(-2, -1),
        tensor_names=("faces", "face_weights"),
        broadcast_compatible=False)
    shape.compare_batch_dimensions(
        tensors=(vertex_attributes, faces, face_weights),
        last_axes=(-3, -3, -2),
        tensor_names=("vertex_attributes", "faces", "face_weights"),
        broadcast_compatible=False)

    asserts.assert_all_above(face_weights, 0)

    batch_dims = faces.shape.ndims - 2
    batch_shape = faces.shape.as_list()[:-2]
    sample_shape = tf.concat(
        (batch_shape, tf.convert_to_tensor(
            value=(num_samples,), dtype=tf.int32)),
        axis=0)

    sample_face_indices = generate_random_face_indices(
        num_samples, face_weights, seed=seed, stateless=stateless)
    sample_vertex_indices = tf.gather(
        faces, sample_face_indices, batch_dims=batch_dims)
    sample_vertices = tf.gather(
        vertex_attributes, sample_vertex_indices, batch_dims=batch_dims)
    barycentric = generate_random_barycentric_coordinates(
        sample_shape,
        dtype=vertex_attributes.dtype,
        seed=seed,
        stateless=stateless)
    barycentric = tf.expand_dims(barycentric, axis=-1)
    sample_points = tf.math.multiply(sample_vertices, barycentric)
    sample_points = tf.reduce_sum(input_tensor=sample_points, axis=-2)
    return sample_points, sample_face_indices


def area_weighted_random_sample_triangle_mesh(
    vertex_attributes: type_alias.TensorLike,
    faces: type_alias.TensorLike,
    num_samples: int,
    vertex_positions: Optional[type_alias.TensorLike] = None,
    seed: Optional[type_alias.TensorLike] = None,
    stateless: bool = False,
    name: str = "area_weighted_random_sample_triangle_mesh"
) -> Tuple[type_alias.TensorLike, type_alias.TensorLike]:
  """Performs a face area weighted random sampling of a tri mesh.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertex_attributes: A `float` tensor of shape `[A1, ..., An, V, D]`, where V
      is the number of vertices, and D is dimensionality of a feature defined on
      each vertex. If `vertex_positions` is not provided, then first 3
      dimensions of `vertex_attributes` denote the vertex positions.
    faces: A `int` tensor of shape `[A1, ..., An, F, 3]`, where F is the number
      of faces.
    num_samples: An `int` scalar denoting number of samples to be drawn from
      each mesh.
    vertex_positions: An optional `float` tensor of shape `[A1, ..., An, V, 3]`,
      where V is the number of vertices. If None, then vertex_attributes[...,
        :3] is used as vertex positions.
    seed: Optional random seed.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then seed must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate same sequence across calls.
    name: Name for op. Defaults to "area_weighted_random_sample_triangle_mesh".

  Returns:
    sample_pts: A `float` tensor of shape `[A1, ..., An, num_samples, D]`,
      where D is dimensionality of each sampled point.
    sample_face_indices: A `int` tensor of shape `[A1, ..., An, num_samples]`.
  """
  with tf.name_scope(name):
    faces = tf.convert_to_tensor(value=faces)
    vertex_attributes = tf.convert_to_tensor(value=vertex_attributes)
    num_samples = tf.convert_to_tensor(value=num_samples)

    shape.check_static(
        tensor=vertex_attributes,
        tensor_name="vertex_attributes",
        has_rank_greater_than=1)
    shape.check_static(
        tensor=vertex_attributes,
        tensor_name="vertex_attributes",
        has_dim_greater_than=(-1, 2))

    if vertex_positions is not None:
      vertex_positions = tf.convert_to_tensor(value=vertex_positions)
    else:
      vertex_positions = vertex_attributes[..., :3]

    shape.check_static(
        tensor=vertex_positions,
        tensor_name="vertex_positions",
        has_rank_greater_than=1)
    shape.check_static(
        tensor=vertex_positions,
        tensor_name="vertex_positions",
        has_dim_equals=(-1, 3))

    triangle_vertex_positions = normals.gather_faces(vertex_positions, faces)
    triangle_areas = triangle_area(triangle_vertex_positions[..., 0, :],
                                   triangle_vertex_positions[..., 1, :],
                                   triangle_vertex_positions[..., 2, :])
    return weighted_random_sample_triangle_mesh(
        vertex_attributes,
        faces,
        num_samples,
        face_weights=triangle_areas,
        seed=seed,
        stateless=stateless)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
