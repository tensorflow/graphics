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
"""Test Cases for mesh refinement layer of Mesh R-CNN."""

from copy import deepcopy

from absl.testing import parameterized

from tensorflow_graphics.projects.mesh_rcnn.layers import test_util
from tensorflow_graphics.projects.mesh_rcnn.layers.mesh_refinement_layer import MeshRefinementLayer
from tensorflow_graphics.util import test_case


class MeshRefinementLayerTest(test_case.TestCase):
  """Tests for whole mesh refinement head."""

  @parameterized.parameters(1, 4)
  def test_refinement_output_shape(self, batch_size):
    """Tests the mesh refinement head on different batch sizes."""

    inputs = test_util.get_mesh_input_data(batch_size, 0, False)

    layer = MeshRefinementLayer(3, 3, 128)
    in_features = inputs['feature']
    in_mesh = deepcopy(inputs['mesh'])
    intrinsics = inputs['intrinsics']

    out_meshes = layer(in_features, in_mesh, intrinsics)

    self.assertNotAllEqual(inputs['mesh'].get_flattened()[0],
                           out_meshes.get_flattened()[0])
    self.assertEqual(inputs['mesh'].get_flattened()[0].shape,
                     out_meshes.get_flattened()[0].shape)
