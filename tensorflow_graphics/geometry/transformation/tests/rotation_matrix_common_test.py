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
"""Tests for rotation_matrix_common."""

from absl.testing import parameterized

from tensorflow_graphics.geometry.transformation import rotation_matrix_common
from tensorflow_graphics.util import test_case


class RotationMatrixCommonTest(test_case.TestCase):

  @parameterized.parameters(
      ((2, 2),),
      ((3, 3),),
      ((4, 4),),
      ((None, 2, 2),),
      ((None, 3, 3),),
      ((None, 4, 4),),
  )
  def test_is_valid_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_not_raised(rotation_matrix_common.is_valid, shape)

  @parameterized.parameters(
      ("must have a rank greater than 1", (2,)),
      ("must have the same number of dimensions in axes", (1, 2)),
      ("must have the same number of dimensions in axes", (None, 2)),
  )
  def test_is_valid_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rotation_matrix_common.is_valid, error_msg,
                                    shape)


if __name__ == "__main__":
  test_case.main()
