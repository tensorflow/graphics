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

  def test_vert_align_with_matching_querypoints(self):
    """Test on set of points, where H & W dimensions are equal to source."""

    image = tf.reshape(tf.range(20.), (1, 4, 5, 1))
    verts = tf.constant([[-1.5, -1.5, 10],
                         [-1.5, 0.5, 10],
                         [-0.5, -1.5, 10],
                         [-0.5, 0.5, 10]], dtype=tf.float32)
    intrinsics = tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                             dtype=tf.float32)

    expected_result = tf.constant([[6.], [8.], [11.], [13.]])
    result = vert_align(image, [verts], intrinsics)

    self.assertAllClose(expected_result, result[0])
