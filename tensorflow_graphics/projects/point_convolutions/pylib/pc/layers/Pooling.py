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
"""Classes for point cloud spatial pooling operations"""

import tensorflow as tf
from pylib.pc.utils import _flatten_features
from pylib.pc.layers.utils import _format_output

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import Neighborhood


class GlobalMaxPooling:
  """ Global max pooling on a point cloud.
  """

  def __call__(self,
               features,
               point_cloud: PointCloud,
               return_padded=False,
               name=None):
    """ Performs a global max pooling on a point cloud.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.

    Args:
      features: A tensor of shape `[N,C]` or `[A1,...,An,V,C]`.
      point_cloud: A `PointCloud` instance.
      return_padded: A `bool`, if `True` reshapes the output to match the
        batch shape of `point_cloud`.

    Returns:
      A tensor of same type as `features` and of shape
        `[B, C]`, if not `return_padded`
      or
        `[A1, ..., An, C]`, if `return_padded`

    """
    features = tf.convert_to_tensor(value=features)
    features = _flatten_features(features, point_cloud)
    features = tf.math.unsorted_segment_max(
        features,
        segment_ids=point_cloud._batch_ids,
        num_segments=point_cloud._batch_size)
    if return_padded:
      shape = tf.concat((point_cloud._batch_shape, [-1]), axis=0)
      features = tf.reshape(features, shape)
    return features


class GlobalAveragePooling:
  """ Global average pooling on a point cloud.
  """

  def __call__(self,
               features,
               point_cloud: PointCloud,
               return_padded=False,
               name=None):
    """ Performs a global average pooling on a point cloud

    Note:
      In the following, `A1` to `An` are optional batch dimensions.

    Args:
      features: A tensor of shape `[N, C]` or `[A1, ..., An, V, C]`.
      point_cloud: A `PointCloud` instance.
      return_padded: A `bool`, if `True` reshapes the output to match the
        batch shape of `point_cloud`.

    Returns:
      A tensor of same type as `features` and of shape
        `[B, C]`, if not `return_padded`
      or
        `[A1 ,..., An, C]`, if `return_padded`

    """
    features = tf.convert_to_tensor(value=features)
    features = _flatten_features(features, point_cloud)
    features = tf.math.unsorted_segment_mean(
        features,
        segment_ids=point_cloud._batch_ids,
        num_segments=point_cloud._batch_size)
    if return_padded:
      shape = tf.concat((point_cloud._batch_shape, [-1]), axis=0)
      features = tf.reshape(features, shape)
    return features


class _LocalPointPooling:
  """ Local point pooling between two point clouds.
  """

  def __call__(self,
               pool_op,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               pooling_radius,
               return_sorted=False,
               return_padded=False,
               name=None,
               default_name="custom pooling"):
    """ Computes a local pooling between two point clouds specified by `pool_op`.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.

    Args:
      pool_op: A function of type `tf.math.unsorted_segmented_*`.
      features: A `float` `Tensor` of shape `[N_in, C]` or
        `[A1, ..., An, V_in, C]`.
      point_cloud_in: A `PointCloud` instance on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output features
        are defined.
      pooling_radius: A `float` or a `float` `Tensor` of shape `[D]`.
      return_sorted: A `bool`, if 'True' the output tensor is sorted
        according to the sorted batch ids of `point_cloud_out`.
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded.

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C]`, if `return_padded` is `True`.

    """
    features = tf.convert_to_tensor(value=features)
    features = _flatten_features(features, point_cloud_in)
    pooling_radius = tf.convert_to_tensor(
        value=pooling_radius, dtype=tf.float32)
    if pooling_radius.shape[0] == 1:
      pooling_radius = tf.repeat(pooling_radius, point_cloud_in._dimension)

    # Compute the grid.
    grid_in = Grid(point_cloud_in, pooling_radius)

    # Compute the neighborhood keys.
    neigh = Neighborhood(grid_in, pooling_radius, point_cloud_out)
    features_on_neighbors = tf.gather(
        features, neigh._original_neigh_ids[:, 0])

    # Pool the features in the neighborhoods
    features_out = pool_op(
        data=features_on_neighbors,
        segment_ids=neigh._original_neigh_ids[:, 1],
        num_segments=tf.shape(point_cloud_out._points)[0])
    return _format_output(features_out,
                          point_cloud_out,
                          return_sorted,
                          return_padded)


class MaxPooling(_LocalPointPooling):
  """ Local max pooling between two point clouds.
  """

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               pooling_radius,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes a local max pooling between two point clouds.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C]` or
        `[A1, ..., An, V_in, C]`.
      point_cloud_in: A `PointCloud` instance on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output features
        are defined.
      pooling_radius: A `float` or a `float` `Tensor` of shape [D].
      return_sorted: A `bool`, if 'True' the output tensor is sorted
        according to the sorted batch ids of `point_cloud_out`.
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded.

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C]`, if `return_padded` is `True`.

    """
    return super(MaxPooling, self).__call__(
        tf.math.unsorted_segment_max,
        features, point_cloud_in, point_cloud_out, pooling_radius,
        return_sorted, return_padded, name, default_name="max pooling")


class AveragePooling(_LocalPointPooling):
  """ Local average pooling between two point clouds.
  """

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               pooling_radius,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes a local average pooling between two point clouds.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C]` or
        `[A1, ..., An, V_in, C]`.
      point_cloud_in: A `PointCloud` instance on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output features
        are defined.
      pooling_radius: A `float` or a `float` `Tensor` of shape `[D]`.
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the sorted batch ids of `point_cloud_out`.
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded.

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C]`, if `return_padded` is `True`.

    """
    return super(AveragePooling, self).__call__(
        tf.math.unsorted_segment_mean,
        features, point_cloud_in, point_cloud_out, pooling_radius,
        return_sorted, return_padded, name, default_name="average pooling")
