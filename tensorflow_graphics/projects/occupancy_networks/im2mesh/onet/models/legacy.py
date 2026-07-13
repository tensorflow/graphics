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

from im2mesh.layers import ResnetBlockFC, AffineLayer


class VoxelDecoder(tf.keras.Model):
  def __init__(self, z_dim=128, c_dim=128, hidden_size=128):
    super().__init__()
    self.c_dim = c_dim
    self.z_dim = z_dim
    # Submodules
    self.actvn = tf.keras.layers.ReLU()
    # 3D decoder
    self.fc_in = tf.keras.layers.Dense(256 * 4 * 4 * 4)
    self.convtrp_0 = tf.keras.layers.Conv3DTranspose(128, 3, stride=2,
                                                     padding='valid', output_padding=1)
    self.convtrp_1 = tf.keras.layers.Conv3DTranspose(64, 3, stride=2,
                                                     padding='valid', output_padding=1)
    self.convtrp_2 = tf.keras.layers.Conv3DTranspose(32, 3, stride=2,
                                                     padding='valid', output_padding=1)
    # Fully connected decoder
    self.z_dim = z_dim
    if not z_dim == 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)

    self.fc_f = tf.keras.layers.Dense(hidden_size)
    self.fc_c = tf.keras.layers.Dense(hidden_size)
    self.fc_p = tf.keras.layers.Dense(hidden_size)

    self.block0 = ResnetBlockFC(hidden_size, hidden_size)
    self.block1 = ResnetBlockFC(hidden_size, hidden_size)
    self.fc_out = tf.keras.layers.Dense(1)

  def call(self, p, z, c):
    batch_size = c.shape[0]

    if self.z_dim != 0:
      net = tf.concat([z, c], axis=1)
    else:
      net = c

    net = self.fc_in(net)
    net = tf.reshape(net, [batch_size, 256, 4, 4, 4])
    net = self.convtrp_0(self.actvn(net))
    net = self.convtrp_1(self.actvn(net))
    net = self.convtrp_2(self.actvn(net))

    """ TODO:
        net = F.grid_sample(
            net, 2*p.unsqueeze(1).unsqueeze(1), padding_mode='border')
    """

    net = tf.transpose(tf.squeeze(
        tf.squeeze(net, axis=2), axis=2), perm=[0, 2, 1])
    net = self.fc_f(self.actvn(net))

    net_p = self.fc_p(p)
    net = net + net_p

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), axis=1)
      net = net + net_z

    if self.c_dim != 0:
      net_c = tf.expand_dims(self.fc_c(c), axis=1)
      net = net + net_c

    net = self.block0(net)
    net = self.block1(net)

    out = self.fc_out(self.actvn(net))
    out = tf.squeeze(out, axis=-1)

    return out


class FeatureDecoder(tf.keras.Model):
  def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=256):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim
    self.dim = dim

    self.actvn = tf.keras.layers.ReLU()

    self.affine = AffineLayer(c_dim, dim)
    if not z_dim == 0:
      self.fc_z = tf.keras.layers.Dense(hidden_size)
    self.fc_p1 = tf.keras.layers.Dense(hidden_size)
    self.fc_p2 = tf.keras.layers.Dense(hidden_size)

    self.fc_c1 = tf.keras.layers.Dense(hidden_size)
    self.fc_c2 = tf.keras.layers.Dense(hidden_size)

    self.block0 = ResnetBlockFC(hidden_size, hidden_size)
    self.block1 = ResnetBlockFC(hidden_size, hidden_size)
    self.block2 = ResnetBlockFC(hidden_size, hidden_size)
    self.block3 = ResnetBlockFC(hidden_size, hidden_size)

    self.fc_out = tf.keras.layers.Dense(1)

  def call(self, p, z, c, **kwargs):
    batch_size, T, D = p.shape

    c1 = tf.reduce_max(tf.reshape(
        c, [batch_size, self.c_dim, -1]), axis=2)[0]
    Ap = self.affine(c1, p)
    Ap2 = Ap[:, :, :2] / (Ap[:, :, 2:].abs() + 1e-5)

    """ TODO
        c2 = F.grid_sample(c, 2*Ap2.unsqueeze(1), padding_mode='border')
        """

    c2 = tf.transpose(tf.squeeze(c2, axis=2), perm=[0, 2, 1])

    net = self.fc_p1(p) + self.fc_p2(Ap)

    if self.z_dim != 0:
      net_z = tf.expand_dims(self.fc_z(z), axis=1)
      net = net + net_z

    net_c = self.fc_c2(c2) + tf.expand_dims(self.fc_c1(c1), axis=1)
    net = net + net_c

    net = self.block0(net)
    net = self.block1(net)
    net = self.block2(net)
    net = self.block3(net)

    out = self.fc_out(self.actvn(net))
    out = tf.squeeze(out, axis=-1)

    return out
