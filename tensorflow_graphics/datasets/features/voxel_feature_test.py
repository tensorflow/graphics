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
# Lint as: python3
"""Tests for tensorflow_graphics.datasets.features.voxel_feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_graphics.datasets.features import voxel_feature

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


class VoxelGridFeatureTest(tfds.testing.FeatureExpectationsTestCase):
  """Test Cases for VoxelGrid FeatureConnector."""

  def test_voxel(self):
    """Tests voxel I/O and encoding/decoding to DatasetFeature."""

    mat_file_path = os.path.join(_TEST_DATA_DIR, 'cube.mat')
    expected_voxel = np.zeros((16, 16, 16), dtype=np.float32)
    expected_voxel[4:12, 4:12, 4:12] = 1.

    mat_dict = {'path': mat_file_path, 'key': 'voxels'}
    raising_inputs = {'path': mat_file_path, 'foo': 'voxels'}
    wrong_key = {'path': mat_file_path, 'key': 'foo'}
    wrong_path = {'path': '/somewhere/wrong', 'key': 'voxels'}
    wrong_dim = np.ones((1, 1, 1, 1))
    self.assertFeature(
        feature=voxel_feature.VoxelGrid((16, 16, 16)),
        shape=(16, 16, 16),
        dtype=tf.float32,
        tests=[
            # mat file
            tfds.testing.FeatureExpectationItem(
                value=mat_dict,
                expected=expected_voxel,
            ),
            # Voxel Grid
            tfds.testing.FeatureExpectationItem(
                value=expected_voxel,
                expected=expected_voxel,
            ),
            tfds.testing.FeatureExpectationItem(
                value=raising_inputs,
                raise_cls=ValueError,
                raise_msg='Missing keys in provided dictionary!',
            ),
            tfds.testing.FeatureExpectationItem(
                value=wrong_key,
                raise_cls=ValueError,
                raise_msg='Key `foo` not found in .mat file',
            ),
            tfds.testing.FeatureExpectationItem(
                value=wrong_path,
                raise_cls=FileNotFoundError,
                raise_msg='File `/somewhere/wrong` does not exist.',
            ),
            tfds.testing.FeatureExpectationItem(
                value=wrong_dim,
                raise_cls=ValueError,
                raise_msg='Only 3D Voxel Grids are supported.',
            ),
        ],
    )


if __name__ == '__main__':
  tfds.testing.test_main()
