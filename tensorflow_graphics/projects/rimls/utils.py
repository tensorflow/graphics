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
"""Utility functions."""
import tensorflow as tf


class NearestNeighbors(object):
  """API class for nearest neighbors."""

  def __init__(self, points, k):
    tf.debugging.assert_less_equal(k, points.shape[0])
    self.k = k
    self.points = points

  def kneighbors(self, queries):
    expanded_queries = tf.expand_dims(queries, 1)
    expanded_points = tf.expand_dims(self.points, 0)
    dist = (expanded_queries - expanded_points) ** 2
    dist = tf.reduce_sum(dist, -1)
    dist = tf.sqrt(dist)
    sorted_indices = tf.argsort(dist, axis=-1)
    return sorted_indices[..., :self.k]
