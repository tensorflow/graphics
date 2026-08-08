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
"""Class to test bounding box"""

import os
from os.path import dirname, abspath
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.io import load_points_from_file_to_numpy
from pylib.io import load_points_from_file_to_tensor
from pylib.io import load_batch_of_points
from pylib.io import load_batch_of_meshes


class IO_test(test_case.TestCase):

  def test_load_points(self):
    data_path = dirname(dirname(dirname(dirname(abspath(__file__))))) + \
         '/test_point_clouds' + '/modelnet40/'
    points_np, features_np = load_points_from_file_to_numpy(
        data_path + 'airplane_0001.txt')
    points_tf, features_tf = load_points_from_file_to_tensor(
        data_path + 'airplane_0001.txt')
    self.assertAllClose(points_np, points_tf)
    self.assertAllClose(features_np, features_tf)

    points_np, features, sizes = load_batch_of_points(
        [data_path + 'airplane_0001.txt'])
    self.assertTrue(points_np.shape == [1, 10000, 3])
    self.assertTrue(sizes.shape == [1])

  def test_load_mesh(self):
    data_path = dirname(dirname(dirname(dirname(abspath(__file__))))) + \
         '/test_point_clouds' + '/meshes/'
    points_np, sizes = load_batch_of_meshes(
        [data_path + '69405.obj'])
    self.assertTrue(points_np.shape == [1, 3960, 3])
    self.assertTrue(sizes.shape == [1])


if __name__ == '__main__':
  test_case.main()
