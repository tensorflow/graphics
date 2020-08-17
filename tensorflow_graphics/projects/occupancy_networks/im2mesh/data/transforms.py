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
""" NO COMMENT NOW"""


import numpy as np


# Transforms
class PointcloudNoise(object):
  ''' Point cloud noise transformation class.

  It adds noise to point cloud data.

  Args:
      stddev (int): standard deviation
  '''

  def __init__(self, stddev):
    self.stddev = stddev

  def __call__(self, data):
    ''' Calls the transformation.

    Args:
        data (dictionary): data dictionary
    '''
    data_out = data.copy()
    points = data[None]
    noise = self.stddev * np.random.randn(*points.shape)
    noise = noise.astype(np.float32)
    data_out[None] = points + noise
    return data_out


class SubsamplePointcloud(object):
  ''' Point cloud subsampling transformation class.

  It subsamples the point cloud data.

  Args:
      N (int): number of points to be subsampled
  '''

  def __init__(self, n):
    self.n = n

  def __call__(self, data):
    ''' Calls the transformation.

    Args:
        data (dict): data dictionary
    '''
    data_out = data.copy()
    points = data[None]
    normals = data['normals']

    indices = np.random.randint(points.shape[0], size=self.n)
    data_out[None] = points[indices, :]
    data_out['normals'] = normals[indices, :]

    return data_out


class SubsamplePoints(object):
  ''' Points subsampling transformation class.

  It subsamples the points data.

  Args:
      N (int): number of points to be subsampled
  '''

  def __init__(self, n):
    self.n = n

  def __call__(self, data):
    ''' Calls the transformation.

    Args:
        data (dictionary): data dictionary
    '''
    points = data[None]
    occ = data['occ']

    data_out = data.copy()
    if isinstance(self.n, int):
      idx = np.random.randint(points.shape[0], size=self.n)
      data_out.update({
          None: points[idx, :],
          'occ': occ[idx],
      })
    else:
      nt_out, nt_in = self.n
      occ_binary = (occ >= 0.5)
      points0 = points[~occ_binary]
      points1 = points[occ_binary]

      idx0 = np.random.randint(points0.shape[0], size=nt_out)
      idx1 = np.random.randint(points1.shape[0], size=nt_in)

      points0 = points0[idx0, :]
      points1 = points1[idx1, :]
      points = np.concatenate([points0, points1], axis=0)

      occ0 = np.zeros(nt_out, dtype=np.float32)
      occ1 = np.ones(nt_in, dtype=np.float32)
      occ = np.concatenate([occ0, occ1], axis=0)

      volume = occ_binary.sum() / len(occ_binary)
      volume = volume.astype(np.float32)

      data_out.update({
          None: points,
          'occ': occ,
          'volume': volume,
      })
    return data_out
