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


class Encoder(tf.keras.Model):
  """ Latent encoder class.

  It encodes the input points and returns mean and standard deviation for the
  posterior Gaussian distribution.

  Args:
      z_dim (int): dimension if output code z
      c_dim (int): dimension of latent conditioned code c
      dim (int): input dimension
      leaky (bool): whether to use leaky ReLUs
  """

  def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim

    # Submodules
    self.fc_pos = tf.keras.layers.Dense(128)

    if c_dim != 0:
      self.fc_c = tf.keras.layers.Dense(128)

    self.fc_0 = tf.keras.layer.Dense(128)
    self.fc_1 = tf.keras.layer.Dense(128)
    self.fc_2 = tf.keras.layer.Dense(128)
    self.fc_3 = tf.keras.layer.Dense(128)
    self.fc_mean = tf.keras.layer.Dense(z_dim)
    self.fc_logstd = tf.keras.layer.Dense(z_dim)

    if not leaky:
      self.actvn = tf.keras.layers.ReLU()
      self.pool = tf.math.reduce_max
    else:
      self.actvn = tf.keras.layers.LeakyReLU(0.2)
      self.pool = tf.math.reduce_mean

  def call(self, p, x, c=None, **kwargs):
    batch_size, T, D = p.get_shape()

    # output size: B x T X F
    net = self.fc_0(tf.expand_dims(x, -1))
    net = net + self.fc_pos(p)

    if self.c_dim != 0:
      net = net + tf.expand_dims(self.fc_c(c), 1)

    net = self.fc_1(self.actvn(net))
    pooled = tf.broadcast_to(
        self.pool(net, axis=1, keepdims=True), net.shape)

    net = tf.concat([net, pooled], axis=2)

    net = self.fc_2(self.actvn(net))
    pooled = tf.broadcast_to(
        self.pool(net, axis=1, keepdims=True), net.shape)
    net = tf.concat([net, pooled], axis=2)

    net = self.fc_3(self.actvn(net))
    # Reduce
    #  to  B x F
    net = self.pool(net, aixs=1)

    mean = self.fc_mean(net)
    logstd = self.fc_logstd(net)

    return mean, logstd
