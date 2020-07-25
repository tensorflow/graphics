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

import os

import numpy as np
from scipy import io as sio
import tensorflow as tf
from tensorflow_datasets import features


class VoxelGrid(features.Tensor):
  """`FeatureConnector` for voxel grids.

  During `_generate_examples`, the feature connector accepts as input any of:

    * `dict`: dictionary containing the path to a {.mat} file and the key under
    which the voxel grid is accessible inside the file.
    Structure of the dictionary:
      {
      'path': 'path/to/file.mat',
      'key': 'foo'
      }

    * `np.ndarray`: float32 numpy array of shape [X,Y,Z] representing the
    voxel grid.

  Output:
    A float32 Tensor with shape [X,Y,Z] containing the voxel occupancies.
  """

  def __init__(self, shape):
    super(VoxelGrid, self).__init__(shape=shape, dtype=tf.float32)

  def encode_example(self, example_data):
    # Path to .mat file
    if isinstance(example_data, dict):

      if not all(key in example_data for key in ['path', 'key']):
        raise ValueError(
            f'Missing keys in provided dictionary! Expecting \'path\''
            f' and \'key\', but {example_data.keys()} were given.')

      if not os.path.exists(example_data['path']):
        raise FileNotFoundError(
            f"File `{example_data['path']}` does not exist.")

      with tf.io.gfile.GFile(example_data['path'], 'rb') as mat_file:
        voxel_mat = sio.loadmat(mat_file)

      if example_data['key'] not in voxel_mat:
        raise ValueError(f"Key `{example_data['key']}` not found in .mat file. "
                         f"Available keys in file: {voxel_mat.keys()}")

      voxel_grid = voxel_mat[example_data['key']].astype(np.float32)

    else:
      if example_data.ndim != 3:
        raise ValueError('Only 3D Voxel Grids are supported.')

      voxel_grid = example_data

    return super(VoxelGrid, self).encode_example(voxel_grid)
