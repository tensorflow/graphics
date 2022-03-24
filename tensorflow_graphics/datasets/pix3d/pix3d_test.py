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
"""Tests for the Pix3D dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow_datasets.public_api as tfds

from tensorflow_graphics.datasets import pix3d


class Pix3dTest(tfds.testing.DatasetBuilderTestCase):
  """Test Cases for Pix3D Dataset implementation."""
  DATASET_CLASS = pix3d.Pix3d
  SPLITS = {
      'train': 2,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  DL_EXTRACT_RESULT = ''
  EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'fakes')
  MOCK_OUT_FORBIDDEN_OS_FUNCTIONS = False
  # SKIP_CHECKSUMS = True

  def setUp(self):  # pylint: disable=invalid-name
    super(Pix3dTest, self).setUp()
    self.builder.TRAIN_SPLIT_IDX = os.path.join(self.EXAMPLE_DIR,
                                                'pix3d_train.npy')
    self.builder.TEST_SPLIT_IDX = os.path.join(self.EXAMPLE_DIR,
                                               'pix3d_test.npy')


if __name__ == '__main__':
  tfds.testing.test_main()
