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
"""
Test cases for vert align Op.
"""

import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.ops.vertex_alignment import \
  vert_align
from tensorflow_graphics.util import test_case


class VertAlignTest(test_case.TestCase):
  """Test Cases for vert align OP."""

  def test_vert_align_without_interpolation(self):
    """Test on set of points, where H & W dimensions are equal to source."""
    N, H, W = 2, 10, 10
    source = tf.random.normal((N, H, W, 3), dtype=tf.float32)
    grid = tf.meshgrid(tf.range(0, limit=W, dtype=tf.int32),
                       tf.range(0, limit=H, dtype=tf.int32))
    batch_query_points = tf.reshape(tf.stack(grid, axis=-1), shape=(-1, 2))
    batch_query_points = tf.concat(
        [batch_query_points, tf.ones((H * W, 1), tf.int32)], axis=1)
    query_points = [batch_query_points] * N

    output = vert_align(source, query_points)
    
    expected_output = tf.reshape(source, (N, -1, 3))

    self.assertAllClose(expected_output, output)
