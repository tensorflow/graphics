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
"""Tests for feature representations."""

from absl.testing import parameterized
from tensorflow_graphics.math import feature_representation
from tensorflow_graphics.util import test_case


class FeatureRepresentationTest(test_case.TestCase):

  @parameterized.parameters(
      (3, (3,)),
      (4, (2, 3)),
      (8, (5, 3, 6)),
  )
  def test_random_rays_exception_exception_not_raised(self,
                                                      num_frequencies,
                                                      *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        feature_representation.positional_encoding, shapes,
        num_frequencies=num_frequencies)

if __name__ == "__main__":
  test_case.main()
