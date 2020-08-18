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
"""Test Cases for Mesh head of Mesh R-CNN."""

from copy import deepcopy

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.branches.mesh_head import \
  MeshRefinementStage, MeshRefinementLayer
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import test_case


def _get_input_data(batch_size=1,
                    vertex_features_dim=128,
                    generate_vert_features=False):
  """Generates valid input data."""
  tf.random.set_seed(42)

  image_features = []
  intrinsics = []
  vertices = []
  faces = []
  for _ in range(batch_size):
    image_features.append(tf.reshape(tf.range(20.), (4, 5, 1)))
    intrinsics.append(tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                                  dtype=tf.float32))
    vertices.append(tf.constant([[-1.5, -1.5, 10],
                                 [-1.5, 0.5, 10],
                                 [-0.5, -1.5, 10],
                                 [-0.5, 0.5, 10]], dtype=tf.float32))
    faces.append(tf.constant([[0, 1, 3], [0, 2, 3]], dtype=tf.int32))

  vert_features = None
  if generate_vert_features:
    vert_features = tf.random.normal((sum([v.shape[0] for v in vertices]),
                                      vertex_features_dim))

  return {
      'feature': tf.stack(image_features, 0),
      'mesh': Meshes(vertices, faces),
      'intrinsics': tf.stack(intrinsics, 0),
      'vertex_features': vert_features
  }


class MeshRefinementStageTest(test_case.TestCase):
  """Tests for one refinement stage."""

  @parameterized.parameters(
      (1, 0, False),
      (1, 128, True),
      (1, 256, True),
      (2, 0, False),
      (2, 128, True),
      (2, 256, True),
  )
  def test_correct_output_shape(self, batch_size, vertex_features_dim,
                                gen_vertex_features):
    """Tests if the layer can be called and returns
    output in the correct shape.
    """

    inputs = _get_input_data(batch_size=batch_size,
                             vertex_features_dim=vertex_features_dim,
                             generate_vert_features=gen_vertex_features)

    stage = MeshRefinementStage(image_features_dim=1,
                                vertex_feature_dim=vertex_features_dim,
                                latent_dim=128,
                                stage_depth=1)

    # deepcopy, so that we can compare if update was applied to mesh geometry.
    output = stage(deepcopy(inputs))

    self.assertIsNotNone(output['vertex_features'])
    self.assertNotAllEqual(inputs['mesh'].get_flattened()[0],
                           output['mesh'].get_flattened()[0])
    self.assertEqual(output['vertex_features'].shape,
                     [batch_size * 4, 128])


class MeshRefinementLayerTest(test_case.TestCase):
  """Tests for whole mesh refinement head."""

  @parameterized.parameters(1, 4)
  def test_refinement_output_shape(self, batch_size):
    """Tests the mesh refinement head on different batch sizes."""

    inputs = _get_input_data(batch_size, 0, False)

    layer = MeshRefinementLayer(3, 3, 128)
    in_features = inputs['feature']
    in_mesh = deepcopy(inputs['mesh'])
    intrinsics = inputs['intrinsics']

    out_meshes = layer(in_features, in_mesh, intrinsics)

    self.assertNotAllEqual(inputs['mesh'].get_flattened()[0],
                           out_meshes.get_flattened()[0])
    self.assertEqual(inputs['mesh'].get_flattened()[0].shape,
                     out_meshes.get_flattened()[0].shape)
