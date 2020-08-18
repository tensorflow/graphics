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
"""Triangle mesh feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow_datasets import features
from tensorflow_graphics.io import triangle_mesh


class TriangleMesh(features.FeaturesDict):
  """`FeatureConnector` for triangle meshes.

  During `_generate_examples`, the feature connector accepts as input any of:

    * `str`: path to a {obj,stl,ply,glb} triangle mesh.
    * `trimesh.Trimesh`: A triangle mesh object.
    * `trimesh.Scene`: A scene object containing multiple TriangleMesh
       objects.
    * `dict:` A dictionary containing the vertices and faces of the mesh (see
       output format below).

  Output:
    A dictionary containing:
    # TODO(b/156112246): Add additional attributes (vertex normals, colors,
    # texture coordinates).

    * 'vertices': A `float32` tensor with shape `[N, 3]` denoting the vertex
    coordinates, where N is the number of vertices in the mesh.
    * 'faces': An `int64` tensor with shape `[F, 3]` denoting the face vertex
    indices, where F is the number of faces in the mesh.

    Note: In case the input specifies a Scene (with multiple meshes), the output
    will be a single TriangleMesh which combines all the triangle meshes in the
    scene.
  """

  def __init__(self):
    super(TriangleMesh, self).__init__({
        'vertices': features.Tensor(shape=(None, 3), dtype=tf.float32),
        'faces': features.Tensor(shape=(None, 3), dtype=tf.uint64),
    })

  def encode_example(self, path_or_trianglemesh):
    """Convert the given triangle mesh into a dict convertible to tf example."""
    if isinstance(path_or_trianglemesh, six.string_types):
      # The parameter is a path.
      with tf.io.gfile.GFile(path_or_trianglemesh, 'rb') as tmesh_file:
        features_dict = self._convert_to_trimesh_feature(
            triangle_mesh.load(tmesh_file))
    elif hasattr(path_or_trianglemesh, 'read') and hasattr(
        path_or_trianglemesh, 'name'):
      # The parameter is a file object.
      features_dict = self._convert_to_trimesh_feature(
          triangle_mesh.load(path_or_trianglemesh))
    elif isinstance(path_or_trianglemesh, dict):
      # The parameter is already a Trimesh dictionary.
      features_dict = path_or_trianglemesh
    else:
      # The parameter is a Trimesh or a Scene.
      features_dict = self._convert_to_trimesh_feature(path_or_trianglemesh)

    return super(TriangleMesh, self).encode_example(features_dict)

  def _convert_to_trimesh_feature(self, obj):
    if isinstance(obj, triangle_mesh.Trimesh):
      vertices = np.array(obj.vertices)
      faces = np.array(obj.faces, dtype=np.uint64)
    elif isinstance(obj, triangle_mesh.Scene):
      # Concatenate all the vertices and faces of the triangle meshes in the
      # scene.
      # TODO(b/156117488): Change to a different merging algorithm to avoid
      # duplicated vertices.
      vertices_list = [
          np.array(mesh.vertices) for mesh in obj.geometry.values()
      ]
      faces_list = np.array([
          np.array(mesh.faces, dtype=np.uint64)
          for mesh in obj.geometry.values()
      ])
      faces_offset = np.cumsum(
          [vertices.shape[0] for vertices in vertices_list], dtype=np.uint64)
      faces_list[1:] += faces_offset[:-1]
      vertices = np.concatenate(vertices_list, axis=0)
      faces = np.concatenate(faces_list, axis=0)
    else:
      raise ValueError('obj should be either a Trimesh or a Scene')
    return {
        'vertices': vertices.astype(np.float32),
        'faces': faces,
    }

  @classmethod
  def from_json_content(cls, value) -> 'TriangleMesh':
    return cls()

  def to_json_content(self):
    return {}
