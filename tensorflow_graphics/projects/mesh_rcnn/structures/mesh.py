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

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.projects.mesh_rcnn.util import padding


class Meshes:
  """Wrapper for batching of Meshes with unequal size."""

  def __init__(self, vertices, faces):
    """
    Contstructs the wrapper object from lists of multiple vertices and faces.

    Vertices and faces are padded and packed into a single tensor each.

    Args:
      vertices: list with N float32 tensors of shape `[V, 3]` containing the
        mesh vertices.
      faces: list with N float32 tensors of shape `[F, 3]` containing the mesh
        faces.
    """
    if len(vertices) != len(faces):
      raise ValueError('Need as many face-lists as vertex-lists.')

    vertices, self.vertex_sizes = padding.pad_list(vertices)
    faces, self.face_sizes = padding.pad_list(faces)

    self.vertices, self.unfold_vertices = utils.flatten_batch_to_2d(vertices, sizes=self.vertex_sizes)
    self.faces, self.unfold_faces = utils.flatten_batch_to_2d(faces, sizes=self.face_sizes)

  def get_flattened(self):
    """
    Returns the flattened vertices and faces.
    Returns:
      A 2D tensor of shape `[N*V, 3]` containing all padded vertices.
      A 2D tensor of shape  `[N*F, 3]` containing all padded faces.
    """
    return self.vertices, self.faces

  def get_padded(self):
    """
    Unpacks vertices and faces and returns them as a padded tensor.
    Returns:
      A tensor of shape `[N, V, 3]` containing the padded vertices and
      a tensor of shape `[N, F, 3]` containing the padded faces.

    """
    return self.unfold_vertices(self.vertices), self.unfold_faces(self.faces)

  def get_unpadded(self):
    """
    Unpads und unstacks all vertices and faces and returns them as lists.

    Returns:
      A list of N vertex tensors of shape `[V',3]`.
      A list of N face tensors of shape `[F',3]`
    """
    vertices, faces = self.get_padded()
    vertices = [vertex[:self.vertex_sizes[i]] for i, vertex in enumerate(vertices)]
    faces = [face[:self.face_sizes[i]] for i, face in enumerate(faces)]

    return vertices, faces
