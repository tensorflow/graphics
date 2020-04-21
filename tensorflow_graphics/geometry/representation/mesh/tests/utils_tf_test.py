#Copyright 2018 Google LLC
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
"""Tests for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
import numpy as np

from tensorflow_graphics.geometry.representation.mesh import utils_tf
from tensorflow_graphics.util import test_case


class UtilsTfTest(test_case.TestCase):

  @parameterized.parameters(
      (np.array(((0, 1, 2),)), [[0, 1], [1, 2], [2, 0]]),
      (np.array(
          ((0, 1, 2), (0, 1, 3))), [[0, 1], [1, 2], [2, 0], [1, 3], [3, 0]]),
  )
  def test_extract_undirected_edges_from_triangular_mesh_preset(
      self, test_inputs, test_outputs):
    """Tests that the output contain the expected edges."""
    num_vertices = tf.math.reduce_max(test_inputs) + 1
    edges = utils_tf.extract_unique_edges_from_triangular_mesh(
        test_inputs, num_vertices, directed_edges=False)
    self.assertAllEqual(self.evaluate(edges), test_outputs)

  @parameterized.parameters(
      (np.array(
          ((0, 1, 2),)), [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]]),
      (np.array(
          ((0, 1, 2), (0, 1, 3))), [[0, 1], [1, 2], [2, 0], [1, 3], [3, 0],
                                    [1, 0], [2, 1], [0, 2], [3, 1], [0, 3]]),
  )
  def test_extract_directed_edges_from_triangular_mesh_preset(
      self, test_inputs, test_outputs):
    """Tests that the output contain the expected edges."""
    num_vertices = tf.math.reduce_max(test_inputs) + 1
    edges = utils_tf.extract_unique_edges_from_triangular_mesh(
        test_inputs, num_vertices, directed_edges=True)
    self.assertAllEqual(self.evaluate(edges), test_outputs)

  @parameterized.parameters(
      (1, "must have a rank equal to 2"),
      (np.array((1,)), "must have a rank equal to 2"),
      (np.array((((1,),),)), "must have a rank equal to 2"),
      (np.array(((1,),)), "must have exactly 3 dimensions in the last axis"),
      (np.array(((1, 1),)), "must have exactly 3 dimensions in the last axis"),
      (np.array(
          ((1, 1, 1, 1),)), "must have exactly 3 dimensions in the last axis"),
  )
  def test_extract_edges_from_triangular_mesh_raised(
      self, invalid_input, error_msg):
    """Tests that the shape exceptions are properly raised."""
    with self.assertRaisesRegexp(ValueError, error_msg):
      num_vertices = tf.math.reduce_max(invalid_input) + 1
      utils_tf.extract_unique_edges_from_triangular_mesh(invalid_input, num_vertices)

  @parameterized.parameters(
      (np.array(((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))),
       tf.float16,
       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
      (np.array(((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))),
       tf.float32,
       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
      (np.array(((0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3),
                 (2, 0), (2, 1), (3, 0), (3, 1))),
       tf.float64,
       [1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3,
        0.5, 0.5, 0.5, 0.5]),
  )
  def test_get_degree_based_edge_weights_preset(
      self, test_inputs, test_dtype, test_outputs):
    """Tests that the output contain the expected edges."""
    weights = utils_tf.get_degree_based_edge_weights(test_inputs, test_dtype)
    self.assertAllClose(self.evaluate(weights), test_outputs)

  @parameterized.parameters(
      (1, "must have a rank equal to 2"),
      (np.array((1,)), "must have a rank equal to 2"),
      (np.array((((1,),),)), "must have a rank equal to 2"),
      (np.array(((1,),)), "must have exactly 2 dimensions in the last axis"),
      (np.array(
          ((1, 1, 1),)), "must have exactly 2 dimensions in the last axis"),
  )
  def test_get_degree_based_edge_weights_invalid_edges_raised(
      self, invalid_input, error_msg):
    """Tests that the shape exceptions are properly raised."""
    with self.assertRaisesRegexp(ValueError, error_msg):
      utils_tf.get_degree_based_edge_weights(invalid_input)

  @parameterized.parameters(
      (tf.bool, "'dtype' must be a tensorflow float type"),
      (tf.int64, "'dtype' must be a tensorflow float type"),
      (tf.complex64, "'dtype' must be a tensorflow float type"),
      (tf.uint64, "'dtype' must be a tensorflow float type"),
      (tf.int16, "'dtype' must be a tensorflow float type"),
  )
  def test_get_degree_based_edge_weights_dtype_raised(
      self, invalid_type, error_msg):
    """Tests that the shape exceptions are properly raised."""
    with self.assertRaisesRegexp(ValueError, error_msg):
      utils_tf.get_degree_based_edge_weights(np.array(((1, 1),)), invalid_type)

if __name__ == "__main__":
  test_case.main()
