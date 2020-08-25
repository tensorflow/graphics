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
"""Data Structure for Meshes"""

import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
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

  def get_flattened(self):
    """
    Returns the flattened vertices and faces.
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
    """
    Unpacks vertices and faces and returns them as a padded tensor.
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
    """
    Unpads und unstacks all vertices and faces and returns them as a flat list.

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

  def _check_valid_input(self, vertices, faces, batch_sizes):
    """
    Checks if the provided input is valid.

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
