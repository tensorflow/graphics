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
"""Classes for network blocks"""

import tensorflow as tf
from pylib.pc.utils import _flatten_features

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import Neighborhood
from pylib.pc import KDEMode

from pylib.pc.layers import MCConv, KPConv, Conv1x1, PointConv
from pylib.pc.layers.utils import _format_output
from pylib.pc.layers.utils import _identity

layer_types = {'mcconv': MCConv,
               'mc': MCConv,
               'mc_conv': MCConv,
               'kpconv': KPConv,
               'kp': KPConv,
               'kp_conv': KPConv,
               'pointconv': PointConv,
               'point_conv': PointConv}


class PointResNet(tf.Module):
  """ ResNet with pre-activation using point cloud convolution layers on one
  point cloud.

  Args:
    num_features: An `int`, the number of features per input point.
    num_blocks: An `int`, the number of Resnet blocks, consisting of 2 layers
      each.
    num_dims: An `int, dimensionality of the point cloud.
    layer_type: A `string`, the type of point cloud convolution layer used.
      Can be `'MCConv'` (default), `'KPConv'`or `'PointConv' `. (optional)
    projection_shortcuts: A `bool`, if `True` a 1x1 convolution is applied to
      the skip connections. Defaults to `False`. (optional)
    activation: A `tf.function`, the activiation used between layers, defaults
      to `tf.nn.relu`. (optional)
    **kwargs: Additional keyword arguments for the convolution layers.
      (optional)

  """

  def __init__(self,
               num_features,
               num_blocks,
               num_dims,
               layer_type='MCConv',
               projection_shortcuts=False,
               activation=tf.nn.relu,
               name=None,
               **kwargs):
    with tf.compat.v1.name_scope(
        name, "Create Monte-Carlo convolution ResNet with pre-activation",
        [num_features, num_blocks, num_dims, layer_type,
         projection_shortcuts, activation, kwargs]):
      self._num_dims = num_dims
      self._num_blocks = num_blocks
      self._activation = activation
      self._projection_shortcuts = projection_shortcuts
      self._batch_norm_layers = []
      self._conv_layers = []
      self._projection_layers = []

      conv = layer_types[layer_type.lower()]
      for i in range(num_blocks):
        # layers inside blocks
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._conv_layers.append(conv(num_features_in=num_features,
                                      num_features_out=num_features,
                                      num_dims=num_dims,
                                      **kwargs))
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._conv_layers.append(conv(num_features_in=num_features,
                                      num_features_out=num_features,
                                      num_dims=num_dims,
                                      **kwargs))
        # layer on skip connection
        if self._projection_shortcuts:
          self._projection_layers.append(Conv1x1(num_features, num_features))
        else:
          self._projection_layers.append(_identity)
        # up- and downwampling in feature domain

  def __call__(self,
               features,
               point_cloud: PointCloud,
               radius,
               training,
               neighborhood=None,
               return_sorted=False,
               return_padded=False,
               name=None,
               **kwargs):
    """ Computes the result of a ResNet with pre-activation using Monte-Carlo
    convolutions.

    Args:
      features: A `float` `Tensor` of shape `[N, C]`.
      point_cloud: A `PointCloud` instance, with `N` points.
      radius: A `float`, the radius used for the convolution.
      training: A `bool`, passed to the batch norm layers.
      neighborhood: A `Neighborhood` instance, defining the neighborhoods with
         `radius` on `point_cloud`.(optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)
      **kwargs: Additional keyword arguments for the convolution layers.
        (optional)

    """
    with tf.compat.v1.name_scope(
        name,
        "Monte-Carlo convolution ResNet with pre-activation",
        [features, point_cloud, radius, training, neighborhood,
         return_sorted, return_padded, kwargs]):
      features = tf.convert_to_tensor(value=features, dtype=tf.float32)
      features = _flatten_features(features, point_cloud)

      if neighborhood is None:

        radii_tensor = tf.repeat([radius], self._num_dims)
        #Compute the grid.
        grid = Grid(point_cloud, radii_tensor)
        #Compute the neighborhood key.
        neighborhood = Neighborhood(grid, radii_tensor)
      for i in range(self._num_blocks):
        skip = features
        # first residual convolution
        features = self._batch_norm_layers[2 * i](features,
                                                  training=training)
        features = self._activation(features)
        features = self._conv_layers[2 * i](features=features,
                                            point_cloud_in=point_cloud,
                                            point_cloud_out=point_cloud,
                                            radius=radius,
                                            neighborhood=neighborhood,
                                            **kwargs)
        # second residual convolution
        features = self._batch_norm_layers[2 * i + 1](features,
                                                      training=training)
        features = self._activation(features)
        features = self._conv_layers[2 * i + 1](features=features,
                                                point_cloud_in=point_cloud,
                                                point_cloud_out=point_cloud,
                                                radius=radius,
                                                neighborhood=neighborhood,
                                                **kwargs)
        # skip connection
        features = features + self._projection_layers[i](skip, point_cloud)
      return _format_output(features,
                            point_cloud,
                            return_sorted,
                            return_padded)


class PointResNetBottleNeck(tf.Module):
  """ ResNet with pre-activation using Monte-Carlo convolution layers on one
  point cloud with a bottle neck in the feature domain.

  Args:
    num_features: An `int`, the number of features per input point.
    bottle_neck_num_features: An `int`, the number of features inside the
      bottle neck blocks. Should be smaller than the `num_features`.
      The feature dimension inside the residual blocks is changed to
      `bottle_neck_num_features` by using 1x1 convolutions for up- and
      downsampling in the feature domain.
    num_blocks: An `int`, the number of Resnet blocks, consisting of 2 layers
      each.
    num_dims: An `int, dimensionality of the point cloud.
    layer_type: A `string`, the type of point cloud convolution layer used.
      Can be `'MCConv'` (default), `'KPConv'`or `'PointConv' `. (optional)
    projection_shortcuts: A `bool`, if `True` a 1x1 convolution is applied to
      the skip connections.
    activation: A `tf.function`, the activiation used between layers, defaults
      to `tf.nn.relu`.
      **kwargs: Additional keyword arguments for the convolution layers.
        (optional)

  """

  def __init__(self,
               num_features,
               bottle_neck_num_features,
               num_blocks,
               num_dims,
               layer_type='MCConv',
               projection_shortcuts=False,
               activation=tf.nn.relu,
               name=None,
               **kwargs):
    with tf.compat.v1.name_scope(
        name, "Create Monte-Carlo convolution ResNet with pre-activation",
        [num_features, bottle_neck_num_features, num_blocks, num_dims,
         layer_type, projection_shortcuts, activation, kwargs]):
      self._num_dims = num_dims
      self._num_blocks = num_blocks
      self._activation = activation
      self._projection_shortcuts = projection_shortcuts

      self._batch_norm_layers = []
      self._conv_layers = []
      self._projection_layers = []
      self._upsampling_layers = []
      self._downsampling_layers = []

      conv = layer_types[layer_type.lower()]
      for i in range(num_blocks):
        # layers inside blocks
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._conv_layers.append(
            conv(num_features_in=bottle_neck_num_features,
                 num_features_out=bottle_neck_num_features,
                 num_dims=num_dims,
                 **kwargs))
        # layer on skip connection
        if self._projection_shortcuts:
          self._projection_layers.append(Conv1x1(num_features, num_features))
        else:
          self._projection_layers.append(_identity)
        # up- and downwampling in feature domain
        self._downsampling_layers.append(Conv1x1(num_features,
                                                 bottle_neck_num_features,
                                                 use_bias=True))
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._upsampling_layers.append(Conv1x1(bottle_neck_num_features,
                                               num_features,
                                               use_bias=True))
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())

  def __call__(self,
               features,
               point_cloud: PointCloud,
               radius,
               training,
               neighborhood=None,
               return_sorted=False,
               return_padded=False,
               name=None,
               **kwargs):
    """ Computes the result of a ResNet with pre-activation using Monte-Carlo
    convolutions.

    Args:
      features: A `float` `Tensor` of shape `[N, C]`.
      point_cloud: A `PointCloud` instance, with `N` points.
      radius: A `float`, the radius used for the convolution.
      training: A `bool`, passed to the batch norm layers.
      neighborhood: A `Neighborhood` instance, defining the neighborhoods with
         `radius` on `point_cloud`.(optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)
      **kwargs: Additional keyword arguments for the convolution layers.
        (optional)

    """
    with tf.compat.v1.name_scope(
        name,
        "Monte-Carlo convolution ResNet with pre-activation",
        [features, point_cloud, radius, training, neighborhood,
         return_sorted, return_padded, kwargs]):
      features = tf.convert_to_tensor(value=features, dtype=tf.float32)
      features = _flatten_features(features, point_cloud)

      if neighborhood is None:
        radii_tensor = tf.repeat([radius], self._num_dims)
        #Compute the grid.
        grid = Grid(point_cloud, radii_tensor)
        #Compute the neighborhood key.
        neighborhood = Neighborhood(grid, radii_tensor)
      for i in range(self._num_blocks):
        skip = features
        # downsampling
        features = self._batch_norm_layers[3 * i](features,
                                                  training=training)
        features = self._activation(features)
        features = self._downsampling_layers[i](features, point_cloud)
        # convolution on downsampled
        features = self._batch_norm_layers[3 * i + 1](features,
                                                      training=training)
        features = self._activation(features)
        features = self._conv_layers[i](features=features,
                                        point_cloud_in=point_cloud,
                                        point_cloud_out=point_cloud,
                                        radius=radius,
                                        neighborhood=neighborhood,
                                        **kwargs)
        # upsampling
        features = self._batch_norm_layers[3 * i + 2](features,
                                                      training=training)
        features = self._activation(features)
        features = self._upsampling_layers[i](features, point_cloud)
        # skip connection
        features = features + self._projection_layers[i](skip, point_cloud)
      return _format_output(features,
                            point_cloud,
                            return_sorted,
                            return_padded)


class PointResNetSpatialBottleNeck(tf.Module):
  """ ResNet with pre-activation using Monte-Carlo convolution layers with
  spatial down- and upsampling.

  The idea of using spatial bottleneck was first proposed by [Accelerating
  Deep Neural Networks with Spatial Bottleneck Modules. Peng et al., 2018]
  (https://arxiv.org/abs/1809.02601)

  Args:
    num_features: An `int`, the number of features per input point.
    num_blocks: An `int`, the number of Resnet blocks, consisting of 4 layers
      each (including down- and upsampling).
    num_dims: An `int, dimensionality of the point cloud.
    layer_type: A `string`, the type of point cloud convolution layer used.
      Can be `'MCConv'` (default), `'KPConv'`or `'PointConv' `. (optional)
    projection_shortcuts: A `bool`, if `True` a 1x1 convolution is applied to
      the skip connections.
    activation: A `tf.function`, the activiation used between layers, defaults
      to `tf.nn.relu`.
    **kwargs: Additional keyword arguments for the convolution layers.
      (optional)
  """

  def __init__(self,
               num_features,
               num_blocks,
               num_dims,
               layer_type='MCConv',
               projection_shortcuts=False,
               activation=tf.nn.relu,
               name=None,
               **kwargs):
    with tf.compat.v1.name_scope(
        name,
        "Create Monte-Carlo convolution ResNetBottleNeck with pre-activation",
        [num_features, num_blocks, num_dims, activation, kwargs]):
      self._num_dims = num_dims
      self._num_blocks = num_blocks
      self._activation = activation
      self._batch_norm_layers = []
      self._conv_layers = []
      self._upsampling_layers = []
      self._downsampling_layers = []
      self._projection_layers = []

      conv = layer_types[layer_type.lower()]
      for i in range(num_blocks):
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._upsampling_layers.append(
            conv(num_features_in=num_features,
                 num_features_out=num_features,
                 num_dims=num_dims,
                 **kwargs))
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._conv_layers.append(
            conv(num_features_in=num_features,
                 num_features_out=num_features,
                 num_dims=num_dims,
                 **kwargs))
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._downsampling_layers.append(
            conv(num_features_in=num_features,
                 num_features_out=num_features,
                 num_dims=num_dims,
                 **kwargs))
        if projection_shortcuts:
          self._projection_layers.append(Conv1x1(num_features,
                                                 num_features))
        else:
          self._projection_layers.append(_identity)

  def __call__(self,
               features,
               point_cloud: PointCloud,
               point_cloud_downsampled: PointCloud,
               conv_radii,
               training,
               neighborhoods=None,
               return_sorted=False,
               return_padded=False,
               name=None,
               **kwargs):
    """ Computes the result of a ResNetBottleNeck with pre-activation using
    Monte-Carlo convolutions.

    Args:
      features: A `float` `Tensor` of shape `[N, C]`.
      point_cloud: A `PointCloud` instance, with `N` points.
      point_cloud_downsampled: A `PointCloud` instance, with the downsampled
        points.
      conv_radii: A `float` `Tensor` of shape `[3]`, the radii used for
          1. spatial downsampling convoltion
          2. convolutions on the downsampled point cloud
          3. spatial upsampling convolution
      training: A `bool`, passed to the batch norm layers.
      neighborhoods: A list of three `Neighborhood` instances, the
        neighborhoods defined between: (optional)
           1. `point_cloud` and `point_cloud_downsampled`
           2. inside `point_cloud_downsampled`
           3. `point_cloud_downsampled` and `point_cloud`
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)
      **kwargs: Additional keyword arguments for the convolution layers.
        (optional)

    """
    with tf.compat.v1.name_scope(
        name,
        "Monte-Carlo convolution ResNetBottleNet with pre-activation",
        [features, point_cloud, point_cloud_downsampled, conv_radii, training,
         neighborhoods, return_sorted, return_padded, kwargs]):
      features = tf.convert_to_tensor(value=features, dtype=tf.float32)
      features = _flatten_features(features, point_cloud)
      conv_radii = tf.convert_to_tensor(value=conv_radii)
      radii = tf.reshape(conv_radii, [3, 1])

      if neighborhoods is None:
        neighborhoods = []
        radii_tensor = tf.repeat(radii, self._num_dims, axis=1)
        # downsampling
        grid_down = Grid(point_cloud, radii_tensor[0])
        neighborhoods.append(Neighborhood(grid_down,
                                          radii_tensor[0],
                                          point_cloud_downsampled))
        # intra-downsampled
        grid = Grid(point_cloud_downsampled, radii_tensor[1])
        neighborhoods.append(Neighborhood(grid,
                                          radii_tensor[1]))
        # upsampling
        if conv_radii[0] == conv_radii[2]:
          neighborhoods.append(neighborhoods[0].transpose())
        else:
          grid_up = Grid(point_cloud_downsampled, radii_tensor[2])
          neighborhoods.append(Neighborhood(grid_up,
                                            radii_tensor[2],
                                            point_cloud))
      for i in range(self._num_blocks):
        skip = features
        features = self._batch_norm_layers[3 * i](features,
                                                  training=training)
        features = self._activation(features)
        features = self._downsampling_layers[i](
            features=features,
            point_cloud_in=point_cloud,
            point_cloud_out=point_cloud_downsampled,
            radius=radii[0],
            neighborhood=neighborhoods[0],
            **kwargs)
        features = self._batch_norm_layers[3 * i + 1](features,
                                                      training=training)
        features = self._activation(features)
        features = self._conv_layers[i](
          features=features,
          point_cloud_in=point_cloud_downsampled,
          point_cloud_out=point_cloud_downsampled,
          radius=radii[1],
          neighborhood=neighborhoods[1],
          **kwargs)
        features = self._batch_norm_layers[3 * i + 2](features,
                                                      training=training)
        features = self._activation(features)
        features = self._upsampling_layers[i](
            features=features,
            point_cloud_in=point_cloud_downsampled,
            point_cloud_out=point_cloud,
            radius=radii[2],
            neighborhood=neighborhoods[2],
            **kwargs)
        features = features + self._projection_layers[i](skip, point_cloud)
      return _format_output(features,
                            point_cloud,
                            return_sorted,
                            return_padded)
