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
"""Test cases for padding util module."""

import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.util.padding import \
  pad_list
from tensorflow_graphics.util import test_case


class PaddingTest(test_case.TestCase):
  """Test cases for padding and stacking of tensors in a list."""

  def test_simple_padding(self):
    """Test with 2 tensors of different shape"""
    values = [tf.ones([1, 4, 3]), tf.ones([1, 2, 3])]
    result, sizes = pad_list(values)
    print(result, sizes)
    self.assertTrue([2, 4, 3] == result.shape)
    self.assertTrue(sizes[1] == 2)

  def test_raises(self):
    """2 Tensors, different rank. Should raise ValueError."""
    values = [tf.ones([1, 4, 3]), tf.ones([1, 2, 2, 3])]
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'All tensors need to have same rank.'):
      _, _ = pad_list(values)

  def test_multiple_paddind_dimensions(self):
    """Tests with 2 tensors of same rank, but padding in more than one
    dimension is requiered. This should raise ValueError."""

    values = [tf.ones([1, 4, 4, 3]), tf.ones([1, 2, 2, 3])]
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Only one dimension with unequal length supported.'):
      _, _ = pad_list(values)

  def test_batch_padd(self):
    """Tests padding for multiple batch dimensions."""
    values = [tf.ones([1, 4, 2, 3]), tf.ones([1, 2, 2, 3])]
    result, sizes = pad_list(values)
    print(result)
    self.assertTrue([2, 4, 2, 3] == result.shape)
    self.assertTrue(sizes[1] == 2)