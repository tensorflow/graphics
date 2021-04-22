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
"""Methods to sample point clouds. """

import tensorflow as tf

from pylib.pc.custom_ops import sampling

from pylib.pc import PointCloud, Neighborhood, Grid
from pylib.pc.utils import cast_to_num_dims

sample_modes = {'average': 1,
                'cell average': 1,
                'cell_average': 1,
                'poisson': 0,
                'poisson disk': 0,
                'poisson_disk': 0}


def sample(neighborhood, sample_mode='poisson', name=None):
  """ Sampling for a neighborhood.

  Args:
    neighborhood: A `Neighborhood` instance.
    sample_mode: A `string`, either `'poisson'`or `'cell average'`.

  Returns:
    A `PointCloud` instance, the sampled points.
    An `int` `Tensor` of shape `[S]`, the indices of the sampled points,
      `None` for cell average sampling.

  """
  with tf.compat.v1.name_scope(
      name, "sample point cloud", [neighborhood, sample_mode]):
    sample_mode_value = sample_modes[sample_mode.lower()]
    #Compute the sampling.
    sampled_points, sampled_batch_ids, sampled_indices = \
        sampling(neighborhood, sample_mode_value)

    #Save the sampled point cloud.
    if sample_mode_value == 0:
      sampled_indices = tf.gather(
          neighborhood._grid._sorted_indices, sampled_indices)
    else:
      sampled_indices = None
    sampled_point_cloud = PointCloud(
        points=sampled_points, batch_ids=sampled_batch_ids,
        batch_size=neighborhood._point_cloud_sampled._batch_size)
    return sampled_point_cloud, sampled_indices
