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
"""Implementation of a shape regularizer based on edge length."""

import tensorflow as tf
from tensorflow_graphics.projects.mesh_rcnn.util import padding
from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import shape


def evaluate(vertices, neighbors, sizes):
  """ Computes an edge loss, which can be used as a shape regularizer for
  learning of high-quality mesh predictions.

  The edge loss is defined as follows:

  $$
    L_{edge}(V, E) = \frac{1}{E} \sum_{(v, v') \in E}||v - v'||^2
  $$

  where `E` represents the set of edges of a mesh.

  Note:
    In the following A1, ..., An are optional batch dimensions.

  Args:
    vertices: A float32 tensor of shape `[A1, ..., An, V, D]` of mesh vertices.
    neighbors: A float32 SparseTensor of shape `[A1, ..., An, V, V]` with
      vertex adjacency information.
    sizes: A int32 tensor of shape `[A1, ..., An]` denoting the sizes of the
      unpacked and unpadded tensors.

  Returns:
    A float32 tensor of shape ´[A1, ..., An]´ storing the edge losses.
  """
  vertices = tf.convert_to_tensor(vertices)

  _check_edge_regularizer_input(vertices, neighbors, sizes)

  edges_per_vertex = tf.sparse.reduce_sum(neighbors, -1)
  edge_sizes = tf.reduce_sum(edges_per_vertex, -1)
  edge_sizes = tf.reshape(edge_sizes, [-1])

  flat_verts, _ = utils.flatten_batch_to_2d(vertices, sizes)
  sizes_squared = tf.stack((sizes, sizes), axis=-1)
  adjacency = utils.convert_to_block_diag_2d(neighbors, sizes=sizes_squared)

  adjacency_ind_0 = adjacency.indices[..., 0]
  adjacency_ind_1 = adjacency.indices[..., 1]

  vertex_features = tf.gather(flat_verts, adjacency_ind_0)
  neighbor_features = tf.gather(flat_verts, adjacency_ind_1)

  difference = vertex_features - neighbor_features
  square_distance = tf.pow(difference, 2.)
  pointwise_square_distance = tf.reduce_sum(square_distance, -1)
  flat_batch_distances = tf.split(pointwise_square_distance,
                                  num_or_size_splits=tf.cast(edge_sizes,
                                                             tf.int32))

  batch_shape = vertices.shape[:-2].as_list()
  padded_distances, n_edges = padding.pad_list(flat_batch_distances)
  full_batched_distances = tf.reshape(padded_distances,
                                      batch_shape + [-1])
  summed_distances = tf.reduce_sum(full_batched_distances, -1)
  n_edges = tf.reshape(n_edges, batch_shape)

  return summed_distances / tf.cast(n_edges, tf.float32)


def _check_edge_regularizer_input(vertices, neighbors, sizes):
  """Checks that the inputs are valid for graph convolution ops.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      data: A float32 tensor with shape `[A1, ..., An, V, D]`.
      neighbors: A SparseTensor with the same type as `data` and with shape
      `[A1, ..., An, V, V]`.
      sizes: An int tensor of shape `[A1, ..., An]`.
    Raises:
      TypeError: if the input types are invalid.
      ValueError: if the input dimensions are invalid.
    """
  if not vertices.dtype.is_floating:
    raise TypeError("'vertices' must have a float type.")
  if neighbors.dtype != vertices.dtype:
    raise TypeError("'neighbors' and 'vertices' must have the same type.")
  if not sizes.dtype.is_integer:
    raise TypeError("'sizes' must have an integer type.")
  if not isinstance(neighbors, tf.sparse.SparseTensor):
    raise ValueError("'neighbors' must be a SparseTensor.")

  vertices_ndims = vertices.shape.ndims
  shape.check_static(tensor=vertices, tensor_name="data",
                     has_rank_greater_than=1)
  shape.check_static(
      tensor=neighbors, tensor_name="neighbors", has_rank=vertices_ndims)

  shape.check_static(
      tensor=sizes, tensor_name="sizes", has_rank=vertices_ndims - 2)
  shape.compare_batch_dimensions(
      tensors=(vertices, neighbors, sizes),
      tensor_names=("data", "neighbors", "sizes"),
      last_axes=(-3, -3, -1),
      broadcast_compatible=False)