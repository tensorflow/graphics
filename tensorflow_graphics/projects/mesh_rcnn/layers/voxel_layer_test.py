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
"""Test cases for the voxel prediction branch of Mesh R-CNN."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.layers import test_util
from tensorflow_graphics.projects.mesh_rcnn.layers.voxel_layer import \
  VoxelPredictionLayer
from tensorflow_graphics.util import test_case


class VoxelPredictionLayerTest(test_case.TestCase):
  """Tests for voxel prediction layer of Mesh R-CNN."""

  @parameterized.parameters(
      (5, 256, 32, 14, 14),
      (0, 128, 16, 28, 28),
      (0, 128, 28, 14, 14),
      (1, 256, 16, 7, 7)
  )
  def test_correct_output_shape(self, num_convs, latent_dim,
                                out_depth, in_width, in_height):
    """Tests the `VoxelPredictionLayer` with different configurations and random
    imput for the correct output shape."""

    # random 4D input tensor: batch_size, in_height, in_width, in_channels.
    input_features = tf.random.normal((2, in_width, in_height, 3))

    layer = VoxelPredictionLayer(num_convs, latent_dim, out_depth)

    output = layer(input_features)

    self.assertEqual(4, tf.rank(output))

    batch_size, height, width = input_features.shape[:3]
    for _ in range(num_convs):
      width, height = test_util.calc_conv_out_spatial_shape(width, height)

    out_width, out_height = test_util.calc_deconv_out_spatial_shape(width,
                                                                    height,
                                                                    2, 2)

    expected_shape = [batch_size, out_height, out_width, out_depth]

    self.assertEqual(expected_shape, output.shape)
