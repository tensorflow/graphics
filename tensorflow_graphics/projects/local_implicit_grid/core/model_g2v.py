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
"""Model for part autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf
layers = tf.keras.layers


class ResBlock3D(layers.Layer):
  """3D convolutional Residue Block.

  Maintains same resolution.
  """

  def __init__(self, neck_channels, out_channels):
    """Initialization.

    Args:
      neck_channels: int, number of channels in bottleneck layer.
      out_channels: int, number of output channels.
    """
    super(ResBlock3D, self).__init__()
    self.neck_channels = neck_channels
    self.out_channels = out_channels
    self.conv1 = layers.Conv3D(neck_channels, kernel_size=1, strides=1)
    self.conv2 = layers.Conv3D(
        neck_channels, kernel_size=3, strides=1, padding="same")
    self.conv3 = layers.Conv3D(out_channels, kernel_size=1, strides=1)
    self.bn1 = layers.BatchNormalization(axis=-1)
    self.bn2 = layers.BatchNormalization(axis=-1)
    self.bn3 = layers.BatchNormalization(axis=-1)

    self.shortcut = layers.Conv3D(out_channels, kernel_size=1, strides=1)

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
    x = tf.nn.relu(x)
    return x


class GridEncoder(layers.Layer):
  """Grid to vector model for part autoencoding."""

  def __init__(self,
               in_grid_res=32,
               num_filters=32,
               codelen=128,
               name="grid_encoder"):
    """Initialization.

    Args:
      in_grid_res: int, input grid resolution, must be powers of 2.
      num_filters: int, number of feature layers at smallest grid resolution.
      codelen: int, length of local latent codes.
      name: str, name of the layer.
    """
    super(GridEncoder, self).__init__(name=name)
    self.in_grid_res = in_grid_res
    self.num_filters = num_filters
    self.codelen = codelen

    # number of input levels.
    self.num_in_level = int(math.log(self.in_grid_res, 2))

    # create layers
    nd = [self.num_filters * (2**i) for i in range(self.num_in_level + 1)
         ]  # feat. in downward path

    self.conv_in = layers.Conv3D(filters=nd[0], kernel_size=1)
    self.down_conv = [ResBlock3D(int(n / 2), n) for n in nd[1:]]
    self.down_pool = [layers.MaxPool3D((2, 2, 2)) for _ in nd[1:]]
    self.fc_out = layers.Dense(self.codelen)

  def call(self, x, training=False):
    """Forward method.

    Args:
      x: `[batch, in_grid_res, in_grid_res, in_grid_res, in_features]` tensor,
        input voxel grid.
      training: bool, flag indicating whether model is in training mode.

    Returns:
      `[batch, codelen]` tensor, output voxel grid.
    """
    x = self.conv_in(x)
    x = tf.nn.relu(x)
    for conv, pool in zip(self.down_conv, self.down_pool):
      x = conv(x, training=training)
      x = pool(x, training=training)  # [batch, res, res, res, c]
    x = tf.squeeze(x, axis=(1, 2, 3))  # [batch, c]
    x = self.fc_out(x)  # [batch, code_len]
    return x


class GridEncoderVAE(layers.Layer):
  """Grid to vector model for part autoencoding."""

  def __init__(self,
               in_grid_res=32,
               num_filters=32,
               codelen=128,
               name="grid_encoder_vae"):
    """Initialization.

    Args:
      in_grid_res: int, input grid resolution, must be powers of 2.
      num_filters: int, number of feature layers at smallest grid resolution.
      codelen: int, length of local latent codes.
      name: str, name of the layer.
    """
    super(GridEncoderVAE, self).__init__(name=name)
    self.in_grid_res = in_grid_res
    self.num_filters = num_filters
    self.codelen = codelen

    # number of input levels
    self.num_in_level = int(math.log(self.in_grid_res, 2))

    # create layers
    # feat. in downward path
    nd = [self.num_filters * (2**i) for i in range(self.num_in_level + 1)]

    self.conv_in = layers.Conv3D(filters=nd[0], kernel_size=1)
    self.down_conv = [ResBlock3D(int(n / 2), n) for n in nd[1:]]
    self.down_pool = layers.MaxPool3D((2, 2, 2))
    self.fc_out = layers.Dense(self.codelen * 2)

  def call(self, x, training=False):
    """Forward method.

    Args:
      x: `[batch, in_grid_res, in_grid_res, in_grid_res, in_features]` tensor,
        input voxel grid.
      training: bool, flag indicating whether model is in training mode.

    Returns:
      `[batch, codelen]` tensor, output voxel grid.
    """
    x = self.conv_in(x)
    x = tf.nn.relu(x)
    for conv in self.down_conv:
      x = conv(x, training=training)
      x = self.down_pool(x, training=training)  # [batch, res, res, res, c]
    x = tf.squeeze(x, axis=(1, 2, 3))  # [batch, c]
    x = self.fc_out(x)  # [batch, code_len*2]
    mu, logvar = x[:, :self.codelen], x[:, self.codelen:]
    noise = tf.random.normal(mu.shape)
    std = tf.exp(0.5 * logvar)
    x_out = mu + noise * std
    return x_out, mu, logvar
