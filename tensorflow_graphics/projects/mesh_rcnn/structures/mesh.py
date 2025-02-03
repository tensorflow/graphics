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
"""Data Structure for Meshes supporting different representations."""

import tensorflow as tf

import tensorflow_graphics.geometry.convolution.utils as utils
from tensorflow_graphics.projects.mesh_rcnn.util import padding


class Meshes:
  """Wrapper for batching of Triangle Meshes."""

  def __init__(self, vertices, faces, batch_sizes=None):
    """Contstructs the wrapper object from nested lists of multiple vertices and
    faces.

    Internally, vertices and faces are padded and packed into a single 2D
    tensor. The object provides accessors for different representations,
    including:

    * List: List of all unpadded vertices and faces tensors. This is intended as
        input format only. However, it is possible to retrieve a flat list of
        all vertices and faces tensor via `get_unpadded()`.
    * Padded: Tensor of shape `[A1, ..., An, max(N), 3]`, where max(N) is the
        size of the largest vertex (or face) list. All other vertex/face tensors
        get padded to this size.
    * Flattened: 2D tensor of shape `[sum(N1, ..., Nn), 3]`, where N1, ..., Nn
        are the sizes of every vertex or face list in this batch.

    Note:
      This implementation supports arbitrary batch dimensions by passing in the
      batch shapes in `batch_sizes`. The vertex and face lists must be flattened
      and reshapeable to the shape defined in `batch_sizes`. E.g. consider the
      case where 4 different meshes are to be stored in a Meshes object, the
      batched shape could be `[2, 2]`, `[4, 1]` or [`1, 4`]. In other words, the
      product over all entries in `batch_sizes` must be equal to the number of
      vertices and faces lists provided in the constructor.

    Args:
      vertices: List with N float32 tensors of shape `[V, 3]` containing the
        mesh vertices, or empty list.
      faces: List with N float32 tensors of shape `[F, 3]` containing the mesh
        faces, or empty list.
      batch_sizes: Optional list of ints indicating the size of each batch
        dimension for the meshes or None, if there is only one batch
        dimension.
    """
    if batch_sizes is not None:
      batch_sizes = tf.convert_to_tensor(batch_sizes)
      # Set batch_sizes to None if only single value was provided, since this is
      # the default behavior.
      batch_sizes = None if tf.size(batch_sizes) == 1 else batch_sizes

    self._check_valid_input(vertices, faces, batch_sizes)

    self.is_empty = len(vertices) == 0

    if not self.is_empty:
      vertices, vertex_sizes = padding.pad_list(vertices)
      faces, face_sizes = padding.pad_list(faces)

      if batch_sizes is not None:
        vert_shape = tf.concat([batch_sizes, vertices.shape[1:]], 0)
        face_shape = tf.concat([batch_sizes, faces.shape[1:]], 0)

        vertices = tf.reshape(vertices, vert_shape)
        faces = tf.reshape(faces, face_shape)

        vertex_sizes = tf.reshape(vertex_sizes, batch_sizes)
        face_sizes = tf.reshape(face_sizes, batch_sizes)

      self.vertex_sizes = vertex_sizes
      self.face_sizes = face_sizes

      self.vertices, self._unfold_vertices = utils.flatten_batch_to_2d(
          vertices, sizes=self.vertex_sizes)
      self.faces, self._unfold_faces = utils.flatten_batch_to_2d(
          faces,
          sizes=self.face_sizes)

      # adjacency will be computed on first call of self.vertex_neighbors()
      self.vertex_adjacency = None

  def get_flattened(self):
    """Returns the flattened vertices and faces.
    Returns:
      A 2D tensor of shape `[N*V, 3]` containing all padded vertices.
      A 2D tensor of shape  `[N*F, 3]` containing all padded faces.
    """
    if self.is_empty:
      vertices = tf.zeros([0, 3], dtype=tf.float32)
      faces = tf.zeros([0, 3], dtype=tf.int32)
    else:
      vertices, faces = self.vertices, self.faces

    return vertices, faces

  def get_padded(self):
    """Unpacks vertices and faces and returns them as a padded tensor.
    Returns:
      A tensor of shape `[N, V, 3]` containing the padded vertices and
      a tensor of shape `[N, F, 3]` containing the padded faces.

    """
    if self.is_empty:
      vertices = tf.zeros([0, 3], dtype=tf.float32)
      faces = tf.zeros([0, 3], dtype=tf.int32)
    else:
      vertices = self._unfold_vertices(self.vertices)
      faces = self._unfold_faces(self.faces)

    return vertices, faces

  def get_unpadded(self):
    """Unpads und unstacks all vertices and faces and returns them as a flat list.

    Returns:
      A list of N vertex tensors of shape `[V',3]`.
      A list of N face tensors of shape `[F',3]`
    """
    vertices, faces = self.get_padded()

    if tf.rank(vertices) < 3:
      return [vertices], [faces]

    vertex_sizes = tf.reshape(self.vertex_sizes, (-1))
    face_sizes = tf.reshape(self.face_sizes, (-1))

    vertices = tf.reshape(vertices, [-1] + vertices.shape[-2:].as_list())
    faces = tf.reshape(faces, [-1] + faces.shape[-2:].as_list())
    vertices = [vertex[:vertex_sizes[i]] for i, vertex in
                enumerate(vertices)]
    faces = [face[:face_sizes[i]] for i, face in enumerate(faces)]

    return vertices, faces

  def get_sizes(self):
    """Return the sizes tensors."""
    return self.vertex_sizes, self.face_sizes

  def add_offsets(self, offsets):
    """
    Adds offsets to mesh vertices, changing the meshes geomtry but
    preserving the topology.

    Args:
      offsets:  float32 tensor of same shape as self.vertices.
    """
    if offsets.shape != self.vertices.shape:
      raise ValueError(
          f'Offsets must be a tensor of shape as {self.vertices.shape}!')

    self.vertices = self.vertices + offsets

    return self

  def _check_valid_input(self, vertices, faces, batch_sizes):
    """Checks if the provided input is valid.

    Args:
      vertices: List of vertices provided in constructor
      faces: List of faces provided in constructor
      batch_sizes: List of tuple of ints provided in constructor or None

    Raises:
      ValueError: if the input is not of a valid formtat.
    """
    if len(vertices) != len(faces):
      raise ValueError('Need as many face-lists as vertex-lists.')

    if batch_sizes is not None:
      if not len(vertices) % tf.reduce_prod(batch_sizes) == 0:
        raise ValueError(f'vertices list of size {len(vertices)} cannot be '
                         f'batched to shape {batch_sizes}!')

      if not len(faces) % tf.reduce_prod(batch_sizes) == 0:
        raise ValueError(f'vertices list of size {len(faces)} cannot be '
                         f'batched to shape {batch_sizes}!')

  def vertex_neighbors(self,
                       return_block_diagonal=False,
                       normalize=False):
    """
    For each vertex i, find all vertices in the 1-ring of vertex i.

    The adjacency matrix will be only computed once, as the meshes are not
    expected to change topology.

    Note:
      In the following, A1 to An are optional batch dimensions that must be
      broadcast compatible.

    Args:
      return_block_diagonal: A boolean flag indicating whether the adjacency
        should be returned as a 2d block-diagonal SparseTensor. If false,
        adjacency will be returned as batch of SparseTensors.
      normalize: A boolean flag indicating whether the adjacency matrix should
        be returned as row normalized matrix, such that
        `adjacency[A1, ..., An, i, j] > 0` if vertex j is a neighbor of i,
        and `adjacency[A1, ..., An, i, i] > 0` for all i, and
        `sum(adjacency, axis=-1)[A1, ..., An, i] == 1.0 for all i`.

    Returns:
      A float32 SparseTensor of shape `[A1, ..., An, D1, D2]`  where
      A1, ..., An are batch dimensions or, `if return_block_diagonal == True`,
      A 2D float32 block-diagonal SparseTensor of shape
      `[A1*...*An*D1, A1*...*An*D2]`.
    """
    if self.vertex_adjacency is None:
      self.vertex_adjacency = self._compute_vertex_adjacency()

    adjacency = self.vertex_adjacency

    if normalize:
      adjacency = self._row_normalize_adjacency_matrix(adjacency)

    if return_block_diagonal:
      sizes = tf.repeat(
          tf.expand_dims(self.vertex_sizes, -1),
          repeats=[2], axis=-1)

      return utils.convert_to_block_diag_2d(adjacency,
                                            sizes=sizes,
                                            validate_indices=True)

    return adjacency

  def _compute_vertex_adjacency(self):
    """For each vertex i, find all vertices in the 1-ring of vertex i.
    This method at first computes all unique bidirectional edges per batch
    instance and builds a SparseTensor representing the adjacency matrix.
    The returned adjacency matrix is represented as a SparseTensor of shape
    `[sum(V1, ..., Vn), sum(V1, ..., Vn)]` where V1, ... Vn represent the
    number of vertices of each mesh in the batch.
    """
    vertices, faces = self.get_padded()
    edges = tf.concat(
        [faces[..., 0:2], faces[..., 1:3], tf.gather(faces, [2, 0], axis=-1)],
        -2)

    batch_shapes = vertices.shape[:-2].as_list()
    n_verts = vertices.shape[-2]
    flat_vertex_sizes = tf.reshape(self.vertex_sizes, [-1])

    # flatten all batch dimensions
    edges_flat = tf.reshape(edges, [-1] + edges.shape[-2:].as_list())

    # hash edges, so that we can use tf.unique, which currently only supports
    # 1D tensors as input.
    edges_hashed = n_verts * edges_flat[..., 0] + edges_flat[..., 1]

    indices = []
    for b_id, batch in enumerate(edges_hashed):
      unique, _ = tf.unique(batch)
      # retrieve vertices from hash value
      edges_padded = tf.stack([unique // n_verts, unique % n_verts], axis=1)
      # padding is now not necessarily at the end of the tensor, hence keep only
      # edges that connect two different vertices
      padding_mask = tf.not_equal(edges_padded[:, 0], edges_padded[:, 1])
      edges = tf.boolean_mask(edges_padded, padding_mask)

      # Also include vertex v into its 1-ring
      batch_verts = tf.range(flat_vertex_sizes[b_id], dtype=tf.int32)
      self_refs = tf.stack([batch_verts, batch_verts], 1)
      index = tf.concat([edges, self_refs], 0)

      # add batch index, so that we can later construct the SparseTensor for the
      # whole batch
      batch_index = tf.expand_dims(
          tf.repeat(tf.constant([b_id]), [len(index)]), 1)
      batch_indices = tf.concat([batch_index, index], -1)
      indices.append(batch_indices)

    sparse_indices = tf.cast(tf.concat(indices, 0), tf.int64)

    size = tf.reduce_prod(batch_shapes)
    if size > 1:
      dense_shape_single_batch = [size, n_verts, n_verts]
    else:
      sparse_indices = sparse_indices[:, 1:]
      dense_shape_single_batch = [n_verts, n_verts]

    neighbors = tf.SparseTensor(
        sparse_indices,
        tf.ones(sparse_indices.shape[0], dtype=tf.float32),
        dense_shape=dense_shape_single_batch
    )
    adjacency = tf.sparse.reorder(neighbors)

    batched_shape = batch_shapes + [n_verts, n_verts]
    adjacency = tf.sparse.reshape(adjacency, batched_shape)

    return adjacency

  def _row_normalize_adjacency_matrix(self, block_diag_2d):
    """Returns a row-normalized version of the adjacency matrix."""
    row_sums = tf.sparse.reduce_sum(block_diag_2d, -1)
    normed = block_diag_2d / tf.repeat(tf.expand_dims(row_sums, -2),
                                       [row_sums.shape[-1]], -2)
    return normed
