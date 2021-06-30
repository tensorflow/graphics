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
"""Tensorflow utility functions to compute normals on meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.representation import triangle
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def gather_faces(vertices: type_alias.TensorLike,
                 indices: type_alias.TensorLike,
                 name: str = "normals_gather_faces"
                 ) -> type_alias.TensorLike:
  """Gather corresponding vertices for each face.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, D]`, where `V` is the number
      of vertices and `D` the dimensionality of each vertex. The rank of this
      tensor should be at least 2.
    indices: A tensor of shape `[A1, ..., An, F, M]`, where `F` is the number of
      faces, and `M` is the number of vertices per face. The rank of this tensor
      should be at least 2.
    name: A name for this op. Defaults to "normals_gather_faces".

  Returns:
    A tensor of shape `[A1, ..., An, F, M, D]` containing the vertices of each
    face.

  Raises:
    ValueError: If the shape of `vertices` or `indices` is not supported.
  """
  with tf.name_scope(name):
    vertices = tf.convert_to_tensor(value=vertices)
    indices = tf.convert_to_tensor(value=indices)

    shape.check_static(
        tensor=vertices, tensor_name="vertices", has_rank_greater_than=1)
    shape.check_static(
        tensor=indices, tensor_name="indices", has_rank_greater_than=1)
    shape.compare_batch_dimensions(
        tensors=(vertices, indices),
        last_axes=(-3, -3),
        broadcast_compatible=False)

    if hasattr(tf, "batch_gather"):
      expanded_vertices = tf.expand_dims(vertices, axis=-3)
      broadcasted_shape = tf.concat(
          [tf.shape(input=indices)[:-1],
           tf.shape(input=vertices)[-2:]],
          axis=-1)
      broadcasted_vertices = tf.broadcast_to(expanded_vertices,
                                             broadcasted_shape)
      return tf.gather(broadcasted_vertices, indices, batch_dims=-1)
    else:
      return tf.gather(
          vertices, indices, axis=-2, batch_dims=indices.shape.ndims - 2)


def face_normals(faces: type_alias.TensorLike,
                 clockwise: bool = True,
                 normalize: bool = True,
                 name: str = "normals_face_normals") -> type_alias.TensorLike:
  """Computes face normals for meshes.

  This function supports planar convex polygon faces. Note that for
  non-triangular faces, this function uses the first 3 vertices of each
  face to calculate the face normal.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    faces: A tensor of shape `[A1, ..., An, M, 3]`, which stores vertices
      positions of each face, where M is the number of vertices of each face.
      The rank of this tensor should be at least 2.
    clockwise: Winding order to determine front-facing faces. The order of
      vertices should be either clockwise or counterclockwise.
    normalize: A `bool` defining whether output normals are normalized.
    name: A name for this op. Defaults to "normals_face_normals".

  Returns:
    A tensor of shape `[A1, ..., An, 3]` containing the face normals.

  Raises:
    ValueError: If the shape of `vertices`, `faces` is not supported.
  """
  with tf.name_scope(name):
    faces = tf.convert_to_tensor(value=faces)

    shape.check_static(
        tensor=faces,
        tensor_name="faces",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3),
        has_dim_greater_than=(-2, 2))

    vertices = tf.unstack(faces, axis=-2)
    vertices = vertices[:3]
    return triangle.normal(*vertices, clockwise=clockwise, normalize=normalize)


def vertex_normals(
    vertices: type_alias.TensorLike,
    indices: type_alias.TensorLike,
    clockwise: bool = True,
    name: str = "normals_vertex_normals") -> type_alias.TensorLike:
  """Computes vertex normals from a mesh.

  This function computes vertex normals as the weighted sum of the adjacent
  face normals, where the weights correspond to the area of each face. This
  function supports planar convex polygon faces. For non-triangular meshes,
  this function converts them into triangular meshes to calculate vertex
  normals.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, 3]`, where V is the number of
      vertices.
    indices: A tensor of shape `[A1, ..., An, F, M]`, where F is the number of
      faces and M is the number of vertices per face.
    clockwise: Winding order to determine front-facing faces. The order of
      vertices should be either clockwise or counterclockwise.
    name: A name for this op. Defaults to "normals_vertex_normals".

  Returns:
    A tensor of shape `[A1, ..., An, V, 3]` containing vertex normals. If
    vertices and indices have different batch dimensions, this function
    broadcasts them into the same batch dimensions and the output batch
    dimensions are the broadcasted.

  Raises:
    ValueError: If the shape of `vertices`, `indices` is not supported.
  """
  with tf.name_scope(name):
    vertices = tf.convert_to_tensor(value=vertices)
    indices = tf.convert_to_tensor(value=indices)

    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=indices,
        tensor_name="indices",
        has_rank_greater_than=1,
        has_dim_greater_than=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(vertices, indices),
        last_axes=(-3, -3),
        broadcast_compatible=True)

    shape_indices = indices.shape.as_list()
    if None in shape_indices[:-2]:
      raise ValueError("'indices' must have specified batch dimensions.")
    common_batch_dims = shape.get_broadcasted_shape(vertices.shape[:-2],
                                                    indices.shape[:-2])
    vertices_repeat = [
        common_batch_dims[x] // vertices.shape.as_list()[x]
        for x in range(len(common_batch_dims))
    ]
    indices_repeat = [
        common_batch_dims[x] // shape_indices[x]
        for x in range(len(common_batch_dims))
    ]
    vertices = tf.tile(
        vertices, vertices_repeat + [1, 1], name="vertices_broadcast")
    indices = tf.tile(
        indices, indices_repeat + [1, 1], name="indices_broadcast")

    # Triangulate non-triangular faces.
    if shape_indices[-1] > 3:
      triangle_indices = []
      for i in range(1, shape_indices[-1] - 1):
        triangle_indices.append(
            tf.concat((indices[..., 0:1], indices[..., i:i + 2]), axis=-1))
      indices = tf.concat(triangle_indices, axis=-2)
      shape_indices = indices.shape.as_list()

    face_vertices = gather_faces(vertices, indices)
    # Use unnormalized face normals to scale normals by area.
    mesh_face_normals = face_normals(
        face_vertices, clockwise=clockwise, normalize=False)

    if vertices.shape.ndims > 2:
      outer_indices = np.meshgrid(
          *[np.arange(i) for i in shape_indices[:-2]],
          sparse=False,
          indexing="ij")
      outer_indices = [np.expand_dims(i, axis=-1) for i in outer_indices]
      outer_indices = np.concatenate(outer_indices, axis=-1)
      outer_indices = np.expand_dims(outer_indices, axis=-2)
      outer_indices = tf.constant(outer_indices, dtype=tf.int32)
      outer_indices = tf.tile(outer_indices, [1] * len(shape_indices[:-2]) +
                              [tf.shape(input=indices)[-2]] + [1])
      unnormalized_vertex_normals = tf.zeros_like(vertices)
      for i in range(shape_indices[-1]):
        scatter_indices = tf.concat([outer_indices, indices[..., i:i + 1]],
                                    axis=-1)
        unnormalized_vertex_normals = tf.tensor_scatter_nd_add(
            unnormalized_vertex_normals, scatter_indices, mesh_face_normals)
    else:
      unnormalized_vertex_normals = tf.zeros_like(vertices)
      for i in range(shape_indices[-1]):
        unnormalized_vertex_normals = tf.tensor_scatter_nd_add(
            unnormalized_vertex_normals, indices[..., i:i + 1],
            mesh_face_normals)

    vector_norms = tf.sqrt(
        tf.reduce_sum(
            input_tensor=unnormalized_vertex_normals**2, axis=-1,
            keepdims=True))
    return safe_ops.safe_unsigned_div(unnormalized_vertex_normals, vector_norms)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
