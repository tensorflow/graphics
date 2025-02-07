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

import tensorflow as tf

from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d,
                            CBatchNorm1d, CBatchNorm1dLegacy, ResnetBlockConv1d)


class Decoder(tf.keras.Model):
  ''' Decoder class.

  It does not perform any form of normalization.

  Args:
      dim (int): input dimension
      z_dim (int): dimension of latent code z
      c_dim (int): dimension of latent conditioned code c
      hidden_size (int): hidden size of Decoder network
      leaky (bool): whether to use leaky ReLUs
  '''

  def __init__(self,
               z_dim=128,
               c_dim=128,
               hidden_size=128,
               leaky=False):  # dim is unnecessary?
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim

    # Submodules
    self.fc_p = tf.keras.layers.Dense(hidden_size)

    if z_dim != 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    if c_dim != 0:
      self.fc_c = tf.keras.layers.Dense(hidden_size)

    # need to check ResnetBlockFC later
    self.block0 = ResnetBlockFC(hidden_size)
    self.block2 = ResnetBlockFC(hidden_size)
    self.block3 = ResnetBlockFC(hidden_size)
    self.block4 = ResnetBlockFC(hidden_size)

    self.fc_out = tf.keras.layers.Dense(1)

    if not leaky:
      self.actvn = tf.keras.layers.ReLU()
    else:
      self.actvn = tf.keras.layers.LeakyReLU(0.2)

  def call(self, p, z, c=None):
    net = self.fc_p(p)

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), 1)
      net = net + net_z

    if self.c_dim != 0:
      net_c = tf.expand_dims(self.fc_c(c), 1)
      net = net + net_c

    net = self.block0(net)
    net = self.block1(net)
    net = self.block2(net)
    net = self.block3(net)
    net = self.block4(net)

    out = self.fc_out(self.actvn(net))
    out = tf.squeeze(out, -1)

    return out


class DecoderCBatchNorm(tf.keras.Model):
  ''' Decoder with conditional batch normalization (CBN) class.

  Args:
      dim (int): input dimension
      z_dim (int): dimension of latent code z
      c_dim (int): dimension of latent conditioned code c
      hidden_size (int): hidden size of Decoder network
      leaky (bool): whether to use leaky ReLUs
      legacy (bool): whether to use the legacy structure
  '''

  def __init__(self,
               z_dim=128,
               c_dim=128,
               hidden_size=256,
               leaky=False,
               legacy=False):
    super().__init__()
    self.z_dim = z_dim
    if z_dim != 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    self.fc_p = tf.keras.layers.Conv1D(hidden_size, 1)

    self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

    if not legacy:
      self.bn = CBatchNorm1d(c_dim, hidden_size)
    else:
      self.bn = CBatchNorm1dLegacy(c_dim, hidden_size)

    self.fc_out = tf.keras.layers.Conv1D(1, 1)

    if not leaky:
      self.actvn = tf.keras.layers.ReLU()
    else:
      self.actvn = tf.keras.layers.LeakyReLU(0.2)

  def call(self, p, z, c, training=False):
    # p = p.transpose(1, 2)
    # p = tf.transpose(p, perm=[0, 2, 1])

    # batch_size, D, T = p.size()
    # batch_size, D, T = p.get_shape() [64,2048,3]
    net = self.fc_p(p)

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), 1)  # CHECK
      net = net + net_z

    net = self.block0(net, c, training=training)
    net = self.block1(net, c, training=training)
    net = self.block2(net, c, training=training)
    net = self.block3(net, c, training=training)
    net = self.block4(net, c, training=training)

    out = self.fc_out(self.actvn(self.bn(net, c, training=training)))

    out = tf.squeeze(out, 2)

    return out


class DecoderCBatchNorm2(tf.keras.Model):
  ''' Decoder with CBN class 2.

  It differs from the previous one in that the number of blocks can be
  chosen.

  Args:
      dim (int): input dimension
      z_dim (int): dimension of latent code z
      c_dim (int): dimension of latent conditioned code c
      hidden_size (int): hidden size of Decoder network
      leaky (bool): whether to use leaky ReLUs
      n_blocks (int): number of ResNet blocks
  '''

  def __init__(self, z_dim=0, c_dim=128, hidden_size=256, n_blocks=5):
    super().__init__()
    self.z_dim = z_dim
    if z_dim != 0:
      self.fc_z = tf.keras.layers.Dense(c_dim)

    self.conv_p = tf.keras.layers.Conv1D(hidden_size, 1)
    self.blocks = [
        CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
    ]  # CHECK nn.ModuleList -> List

    self.bn = CBatchNorm1d(c_dim, hidden_size)
    self.conv_net = tf.keras.layers.Conv1D(1, 1)
    self.actvn = tf.keras.layers.ReLU()

  def call(self, p, z, c, training=False):
    p = tf.transpose(p, perm=[0, 2, 1])
    net = self.conv_p(p)

    if self.z_dim != 0:
      c = c + self.fc_z(z)

    for block in self.blocks:
      net = block(net, c, training=training)

    out = self.conv_out(self.actvn(self.bn(net, c, training=training)))
    out = tf.square(out, 1)

    return out


class DecoderCBatchNormNoResnet(tf.keras.Model):
  ''' Decoder CBN with no ResNet blocks class.

  Args:
      dim (int): input dimension
      z_dim (int): dimension of latent code z
      c_dim (int): dimension of latent conditioned code c
      hidden_size (int): hidden size of Decoder network
      leaky (bool): whether to use leaky ReLUs
  '''

  def __init__(self,
               z_dim=128,
               c_dim=128,
               hidden_size=256,
               leaky=False):
    super().__init__()
    self.z_dim = z_dim
    if z_dim != 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    self.fc_p = tf.keras.layers.Conv1D(hidden_size, 1)
    self.fc_0 = tf.keras.layers.Conv1D(hidden_size)
    self.fc_1 = tf.keras.layers.Conv1D(hidden_size)
    self.fc_2 = tf.keras.layers.Conv1D(hidden_size)
    self.fc_3 = tf.keras.layers.Conv1D(hidden_size)
    self.fc_4 = tf.keras.layers.Conv1D(hidden_size)

    self.bn_0 = CBatchNorm1d(c_dim, hidden_size)
    self.bn_1 = CBatchNorm1d(c_dim, hidden_size)
    self.bn_2 = CBatchNorm1d(c_dim, hidden_size)
    self.bn_3 = CBatchNorm1d(c_dim, hidden_size)
    self.bn_4 = CBatchNorm1d(c_dim, hidden_size)
    self.bn_5 = CBatchNorm1d(c_dim, hidden_size)

    self.fc_out = tf.keras.layers.Dense(1, 1)

    if not leaky:
      self.actvn = tf.keras.layers.ReLU()
    else:
      self.actvn = tf.keras.layers.LeakyReLU(0.2)

  def call(self, p, z, c, training=False):
    p = tf.transpose(p, perm=[0, 2, 1])
    net = self.fc_p(p)

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), 2)
      net = net + net_z

    net = self.actvn(self.bn_0(net, c, training=training))
    net = self.fc_0(net)
    net = self.actvn(self.bn_1(net, c, training=training))
    net = self.fc_1(net)
    net = self.actvn(self.bn_2(net, c, training=training))
    net = self.fc_2(net)
    net = self.actvn(self.bn_3(net, c, training=training))
    net = self.fc_3(net)
    net = self.actvn(self.bn_4(net, c, training=training))
    net = self.fc_4(net)
    net = self.actvn(self.bn_5(net, c, training=training))
    out = self.fc_out(net)
    out = tf.squeeze(out, 1)

    return out


class DecoderBatchNorm(tf.keras.Model):
  ''' Decoder with batch normalization class.

  Args:
      dim (int): input dimension
      z_dim (int): dimension of latent code z
      c_dim (int): dimension of latent conditioned code c
      hidden_size (int): hidden size of Decoder network
      leaky (bool): whether to use leaky ReLUs
  '''

  def __init__(self,
               z_dim=128,
               c_dim=128,
               hidden_size=256,
               leaky=False):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim

    # Submodules
    if z_dim != 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    if self.c_dim != 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    self.fc_p = tf.keras.layers.Conv1D(hidden_size, 1)
    self.block0 = ResnetBlockConv1d(hidden_size)
    self.block1 = ResnetBlockConv1d(hidden_size)
    self.block2 = ResnetBlockConv1d(hidden_size)
    self.block3 = ResnetBlockConv1d(hidden_size)
    self.block4 = ResnetBlockConv1d(hidden_size)

    self.bn = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    self.fc_out = tf.keras.Conv1D(1, 1)

    if not leaky:
      self.actvn = tf.keras.layers.ReLU()
    else:
      self.actvn = tf.keras.layers.LeakyReLU(0.2)

  def call(self, p, z, c, training=False):
    p = tf.transpose(p, perm=[0, 2, 1])
    net = self.fc_p(p)

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), 2)  # CHECK
      net = net + net_z

    if self.c_dim != 0:
      net_c = tf.expand_dims(self.fc_c(c), 2)  # CHECK
      net = net + net_c

    net = self.block0(net, training=training)
    net = self.block1(net, training=training)
    net = self.block2(net, training=training)
    net = self.block3(net, training=training)
    net = self.block4(net, training=training)

    out = self.fc_out(self.actvn(self.bn(net, training=training)))
    out = tf.squeeze(out, 1)

    return out
