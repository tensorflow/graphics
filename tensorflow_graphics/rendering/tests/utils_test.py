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
"""Tests for common util functions."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.rendering import utils
from tensorflow_graphics.util import test_case


class UtilsTest(test_case.TestCase):

  @parameterized.named_parameters(
      ('non-batched xyz', False),
      ('batched xyz', True),
  )
  def test_transform_homogeneous_shapes(self, do_batched):
    num_vertices = 10
    batch_size = 3
    num_channels = 3
    vertices_shape = ([batch_size, num_vertices, num_channels]
                      if do_batched else [num_vertices, num_channels])
    vertices = tf.ones(vertices_shape, dtype=tf.float32)
    matrices = tf.eye(4, dtype=tf.float32)
    if do_batched:
      matrices = tf.tile(matrices[tf.newaxis, ...], [batch_size, 1, 1])

    transformed = utils.transform_homogeneous(matrices, vertices)

    expected_shape = ([batch_size, num_vertices, 4]
                      if do_batched else [num_vertices, 4])
    self.assertEqual(transformed.shape, expected_shape)
