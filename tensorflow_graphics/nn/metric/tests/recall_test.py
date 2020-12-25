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
"""Tests for the recall metric."""

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.nn.metric import recall
from tensorflow_graphics.util import test_case


def random_tensor(tensor_shape):
  return np.random.uniform(low=0.0, high=1.0, size=tensor_shape)


def random_tensor_shape():
  tensor_size = np.random.randint(5) + 1
  return np.random.randint(1, 10, size=(tensor_size)).tolist()


class RecallTest(test_case.TestCase):

  @parameterized.parameters(
      # recall = 0.25.
      ((0, 1, 1, 1, 1), (1, 1, 0, 0, 0), 0.25),
      # recall = 1.
      ((0, 0, 0, 1, 1, 1, 0, 1), (0, 0, 0, 1, 1, 1, 0, 1), 1),
      # All-0 predictions, returns 0.
      ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), 0),
      # All-0 ground truth, returns 0.
      ((0, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 1), 0),
  )
  def test_evaluate_preset(self, ground_truth, predictions, expected_recall):
    tensor_shape = random_tensor_shape()

    ground_truth_labels = np.tile(ground_truth, tensor_shape + [1])
    predicted_labels = np.tile(predictions, tensor_shape + [1])
    expected = np.tile(expected_recall, tensor_shape)

    result = recall.evaluate(ground_truth_labels, predicted_labels, classes=[1])

    self.assertAllClose(expected, result)

  @parameterized.parameters(
      # Recall for classes 2, 3: [1/3, 1.]
      ((2, 0, 3, 1, 2, 2), (2, 3, 3, 2, 3, 3), [2, 3], False, [1. / 3, 1]),
      # Average recall for classes 2, 3: 2/3
      ((2, 0, 3, 1, 2, 2), (2, 3, 3, 2, 3, 3), [2, 3], True, 2. / 3),
      # Recall for all classes: [0, 0, 0.5, 0.5]
      ((1, 2, 3, 3, 1, 1, 2),
       (0, 2, 0, 3, 0, 2, 0), None, False, [0, 0, 0.5, 0.5]),
      # Average recall for all classes: 0.25
      ((1, 2, 3, 3, 1, 1, 2), (0, 2, 0, 3, 0, 2, 0), None, True, 0.25),
  )
  def test_evaluate_preset_multiclass(self, ground_truth, predictions, classes,
                                      reduce_average, expected_recall):
    tensor_shape = random_tensor_shape()

    ground_truth_labels = np.tile(ground_truth, tensor_shape + [1])
    predicted_labels = np.tile(predictions, tensor_shape + [1])
    expected = np.tile(expected_recall,
                       tensor_shape + ([1] if not reduce_average else []))

    result = recall.evaluate(ground_truth_labels, predicted_labels, classes,
                             reduce_average)

    self.assertAllClose(expected, result)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (1, 5, 3), (4, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 4), (2, 4, 5)),
  )
  def test_evaluate_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(recall.evaluate, error_msg, shape)

  @parameterized.parameters(
      ((1, 5, 3), (2, 5, 1)),
      ((None, 2, 6), (4, 2, None)),
      ((3, 1, 1, 2), (3, 5, 8, 2)),
  )
  def test_evaluate_shape_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(recall.evaluate, shapes)


if __name__ == "__main__":
  test_case.main()
