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
"""Test Cases for mesh refinement stage of Mesh R-CNN."""

from copy import deepcopy

from absl.testing import parameterized

from tensorflow_graphics.projects.mesh_rcnn.layers import test_util
from tensorflow_graphics.projects.mesh_rcnn.layers.mesh_refinement_stage import MeshRefinementStage
from tensorflow_graphics.util import test_case


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

    inputs = test_util.get_mesh_input_data(batch_size=batch_size,
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
