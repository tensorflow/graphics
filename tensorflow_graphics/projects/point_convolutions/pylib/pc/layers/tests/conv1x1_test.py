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
# See the License for the specific
"""Class to test pooling layers"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, Grid, Neighborhood, AABB
from pylib.pc.tests import utils
from pylib.pc.layers import Conv1x1


class Conv1x1Test(test_case.TestCase):

  @parameterized.parameters(
    (1000, 4, [3, 3], 3),
    (1000, 4, [3, 1], 3),
    (1000, 4, [1, 3], 3),
  )
  def test_conv1x1(self, num_points, batch_size, feature_sizes, dimension):
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        equal_sized_batches=True)
    features = np.random.rand(batch_size, num_points, feature_sizes[0])
    point_cloud = PointCloud(points, batch_ids)

    conv_layer = Conv1x1(feature_sizes[0], feature_sizes[1])
    result = conv_layer(features, point_cloud)
    self.assertTrue(result.shape[-1] == feature_sizes[1])


if __name__ == '__main___':
  test_case.main()
