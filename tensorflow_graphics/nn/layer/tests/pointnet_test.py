#Copyright 2019 Google LLC
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
"""Tests for pointnet layers."""


from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import test_case
from tensorflow_graphics.nn.layer.pointnet import VanillaEncoder

class PointNetTest(test_case.TestCase):
  
  # @parameterized.parameters()
  def test_forward_features(self):
    points_batch = tf.random.uniform((32, 2048, 3))
    encoder = VanillaEncoder(2048)
    encoder(points_batch, training=False)

    # self.assertAllClose(u_new, u)

if __name__ == "__main__":
  test_case.main()
