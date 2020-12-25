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
"""Model for voxel grid-to-grid encoding."""

import math
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.local_implicit_grid.core import local_implicit_grid_layer as lig

layers = tf.keras.layers


class ResBlock3D(layers.Layer):
  """3D convolutional Residue Block.

  Maintains same resolution.
  """

  def __init__(self, neck_channels, out_channels, final_relu=True):
    """Initialization.

    Args:
      neck_channels: int, number of channels in bottleneck layer.
      out_channels: int, number of output channels.
      final_relu: bool, add relu to the last layer.
    """
    super(ResBlock3D, self).__init__()
    self.neck_channels = neck_channels
    self.out_channels = out_channels
    self.conv1 = layers.Conv3D(neck_channels, kernel_size=1, strides=1)
    self.conv2 = layers.Conv3D(
        neck_channels, kernel_size=3, strides=1, padding='same')
    self.conv3 = layers.Conv3D(out_channels, kernel_size=1, strides=1)
    self.bn1 = layers.BatchNormalization(axis=-1)
    self.bn2 = layers.BatchNormalization(axis=-1)
    self.bn3 = layers.BatchNormalization(axis=-1)

    self.shortcut = layers.Conv3D(out_channels, kernel_size=1, strides=1)
    self.final_relu = final_relu

  def call(self, x, training=False):
    identity = x

    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x += self.shortcut(identity)
    if self.final_relu:
      x = tf.nn.relu(x)

    return x


class UNet3D(tf.keras.layers.Layer):
  """UNet that consumes even dimension grid and outputs even dimension grid."""

  def __init__(self,
               in_grid_res=32,
               out_grid_res=16,
               num_filters=16,
               max_filters=512,
               out_features=32,
               name='unet3d'):
    """Initialization.

    Args:
      in_grid_res: int, input grid resolution, must be powers of 2.
      out_grid_res: int, output grid resolution, must be powers of 2.
      num_filters: int, number of feature layers at smallest grid resolution.
      max_filters: int, max number of feature layers at any resolution.
      out_features: int, number of output feature channels.
      name: str, name of the layer.

    Raises:
      ValueError: if in_grid_res or out_grid_res is not powers of 2.
    """
    super(UNet3D, self).__init__(name=name)
    self.in_grid_res = in_grid_res
    self.out_grid_res = out_grid_res
    self.num_filters = num_filters
    self.max_filters = max_filters
    self.out_features = out_features

    # assert dimensions acceptable
    if math.log(out_grid_res, 2) % 1 != 0 or math.log(in_grid_res, 2) % 1 != 0:
      raise ValueError('in_grid_res and out_grid_res must be 2**n.')

    self.num_in_level = math.log(self.in_grid_res, 2)
    self.num_out_level = math.log(self.out_grid_res, 2)
    self.num_in_level = int(self.num_in_level)  # number of input levels
    self.num_out_level = int(self.num_out_level)  # number of output levels

    self._create_layers()

  def _create_layers(self):
    num_filter_down = [
        self.num_filters * (2**(i + 1)) for i in range(self.num_in_level)
    ]
    # num. features in downward path
    num_filter_down = [
        n if n <= self.max_filters else self.max_filters
        for n in num_filter_down
    ]
    num_filter_up = num_filter_down[::-1][:self.num_out_level]
    self.num_filter_down = num_filter_down
    self.num_filter_up = num_filter_up
    self.conv_in = ResBlock3D(self.num_filters, self.num_filters)
    self.conv_out = ResBlock3D(
        self.out_features, self.out_features, final_relu=False)
    self.down_modules = [ResBlock3D(int(n / 2), n) for n in num_filter_down]
    self.up_modules = [ResBlock3D(n, n) for n in num_filter_up]
    self.dnpool = layers.MaxPool3D((2, 2, 2))
    self.upsamp = layers.UpSampling3D((2, 2, 2))
    self.up_final = layers.UpSampling3D((2, 2, 2))

  def call(self, x, training=False):
    """Forward method.

    Args:
      x: `[batch, in_grid_res, in_grid_res, in_grid_res, in_features]` tensor,
        input voxel grid.
      training: bool, flag indicating whether model is in training mode.

    Returns:
      `[batch, out_grid_res, out_grid_res, out_grid_res, out_features]` tensor,
      output voxel grid.
    """
    x = self.conv_in(x)
    x_dns = [x]
    for mod in self.down_modules:
      x_ = self.dnpool(mod(x_dns[-1], training=training))
      x_dns.append(x_)

    x_ups = [x_dns.pop(-1)]
    for mod in self.up_modules:
      x_ = tf.concat([self.upsamp(x_ups[-1]), x_dns.pop(-1)], axis=-1)
      x_ = mod(x_, training=training)
      x_ups.append(x_)

    x = self.conv_out(x_ups[-1])
    return x


class UNet3DOdd(tf.keras.layers.Layer):
  """UNet that consumes even dimension grid and outputs odd dimension grid."""

  def __init__(self,
               in_grid_res=32,
               out_grid_res=15,
               num_filters=16,
               max_filters=512,
               out_features=32,
               name='unet3dodd'):
    """Initialization.

    Args:
      in_grid_res: int, input grid resolution, must be powers of 2.
      out_grid_res: int, output grid resolution, must be powers of 2.
      num_filters: int, number of feature layers at smallest grid resolution.
      max_filters: int, max number of feature layers at any resolution.
      out_features: int, number of output feature channels.
      name: str, name of the layer.

    Raises:
      ValueError: if in_grid_res or out_grid_res are not 2**n or 2**n-1 for
        some positive integer n.
    """
    super(UNet3DOdd, self).__init__(name=name)
    self.in_grid_res = in_grid_res
    self.out_grid_res = out_grid_res
    self.num_filters = num_filters
    self.max_filters = max_filters
    self.out_features = out_features
    # assert dimensions acceptable
    if math.log(out_grid_res + 1, 2) % 1 != 0 or math.log(in_grid_res,
                                                          2) % 1 != 0:
      raise ValueError(
          'in_grid_res must be 2**n, out_grid_res must be 2**n-1.')
    self.num_in_level = math.log(self.in_grid_res, 2)
    self.num_out_level = math.log(self.out_grid_res + 1, 2)
    self.num_in_level = int(self.num_in_level)  # number of input levels
    self.num_out_level = int(self.num_out_level)  # number of output levels

    self._create_layers()

  def _create_layers(self):
    num_filter_down = [
        self.num_filters * (2**(i + 1)) for i in range(self.num_in_level)
    ]
    # num. features in downward path
    num_filter_down = [
        n if n <= self.max_filters else self.max_filters
        for n in num_filter_down
    ]
    num_filter_up = num_filter_down[::-1][1:self.num_out_level]
    self.num_filter_down = num_filter_down
    self.num_filter_up = num_filter_up
    self.conv_in = ResBlock3D(self.num_filters, self.num_filters)
    self.conv_out = ResBlock3D(
        self.out_features, self.out_features, final_relu=False)
    self.down_modules = [ResBlock3D(int(n / 2), n) for n in num_filter_down]
    self.up_modules = [ResBlock3D(n, n) for n in num_filter_up]
    self.dnpool = layers.MaxPool3D((2, 2, 2))
    self.upsamp = layers.UpSampling3D((2, 2, 2))
    self.up_final = layers.UpSampling3D((2, 2, 2))

  def call(self, x, training=False):
    """Forward method.

    Args:
      x: `[batch, in_grid_res, in_grid_res, in_grid_res, in_features]` tensor,
        input voxel grid.
      training: bool, flag indicating whether model is in training mode.

    Returns:
      `[batch, out_grid_res, out_grid_res, out_grid_res, out_features]` tensor,
      output voxel grid.
    """
    x = self.conv_in(x)
    x_dns = [x]
    for mod in self.down_modules:
      x_ = self.dnpool(mod(x_dns[-1], training=training))
      x_dns.append(x_)

    x_ups = [x_dns.pop(-1)]
    for mod in self.up_modules:
      x_ = tf.concat([self.upsamp(x_ups[-1]), x_dns.pop(-1)], axis=-1)
      x_ = mod(x_, training=training)
      x_ups.append(x_)
    # odd layer
    x = self.upsamp(x_ups[-1])[:, :-1, :-1, :-1, :]
    x = self.conv_out(x)
    return x


class ModelG2G(tf.keras.Model):
  """Grid-to-Grid Model with U-Net skip connections."""

  def __init__(self,
               in_grid_res=32,
               out_grid_res=8,
               num_filters=256,
               codelen=128,
               out_features=1,
               net_type='imnet',
               name='g2g'):
    """Initialization.

    Args:
      in_grid_res: int, input grid resolution, must be powers of 2.
      out_grid_res: int, output grid resolution, must be powers of 2.
      num_filters: int, number of feature layers at smallest grid resolution.
      codelen: int, length of local latent codes.
      out_features: int, number of output feature channels.
      net_type: str, implicit function network architecture. imnet/deepsdf.
      name: str, name of the layer.

    Raises:
      NotImplementedError: if net_type is not imnet or deepsdf.
      ValueError: if in_grid_res or out_grid_res is not powers of 2.
    """
    super(ModelG2G, self).__init__(name=name)
    if math.log(out_grid_res, 2) % 1 != 0 or math.log(in_grid_res, 2) % 1 != 0:
      raise ValueError('in_grid_res and out_grid_res must be powers of 2.')
    if net_type not in ['imnet', 'deepsdf']:
      raise NotImplementedError
    self.codelen = codelen
    self.out_features = out_features
    self.in_grid_res = in_grid_res
    self.out_grid_res = out_grid_res
    self.num_filters = num_filters
    self.net_type = net_type

    self.outgrid = None

    self.unet = UNet3D(
        in_grid_res=in_grid_res,
        out_grid_res=out_grid_res,
        num_filters=num_filters,
        out_features=codelen)
    self.lig = lig.LocalImplicitGrid(
        size=(out_grid_res, out_grid_res, out_grid_res),
        in_features=codelen,
        out_features=out_features,
        net_type=net_type)

  def call(self, voxgrid, pts, training=False):
    """Forward method.

    Args:
      voxgrid: `[batch, inres, inres, inres, nc]` tensor, input voxel grid.
      pts: `[batch, num_points, 3]` tensor, input query points.
      training: bool, flag indicating whether model is in training mode.
    Returns:
      vals: `[batch, num_points, 3]` tensor, predicted values at query points.
    """
    self.outgrid = self.unet(voxgrid, training=training)
    val = self.lig(self.outgrid, pts, training=training)
    return val
