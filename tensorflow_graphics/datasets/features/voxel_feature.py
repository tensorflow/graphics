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
"""Voxel grid feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.io import loadmat
import tensorflow.compat.v2 as tf

from tensorflow_datasets import features


class VoxelGrid(features.Tensor):
  """`FeatureConnector` for voxel grids.

  During `_generate_examples`, the feature connector accepts as input any of:

    * `dict`: dictionary containing the path to a {.mat} file and the key under which the voxel grid
    is accessible inside the file. Structure of the dictionary:
      {
      'path': 'path/to/file.mat',
      'key': 'foo'
      }
    * `np.ndarray`: A voxel grid as numpy array.

  Output:
    A float32 Tensor with shape [X,Y,Z]containing the voxels occupancies.
  """

  def __init__(self, shape):
    super(VoxelGrid, self).__init__(shape=shape, dtype=tf.float32)

  def encode_example(self, example_data):
    # Path to .mat file
    if isinstance(example_data, dict):
      voxels = loadmat(example_data['path'])[example_data['key']].astype(np.float32)
    else:
      if not example_data.ndim == 3:
        raise ValueError("Only 3D Voxel Grids are supported.")
      voxels = example_data

    return super(VoxelGrid, self).encode_example(voxels)
