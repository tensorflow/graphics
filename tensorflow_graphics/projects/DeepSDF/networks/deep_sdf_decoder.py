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
import tensorflow_addons as tfa


class Decoder(tf.keras.Model):
  """ DeepSDF decoder module. """

  def __init__(
      self,
      latent_size,
      dims,
      dropout=None,
      dropout_prob=0.0,
      norm_layers=(),
      latent_in=(),
      weight_norm=False,
      xyz_in_all=None,
      use_tanh=False,
      latent_dropout=False,
  ):
    super(Decoder, self).__init__()

    dims = [latent_size + 3] + dims + [1]

    self.num_layers = len(dims)
    self.norm_layers = norm_layers
    self.latent_in = latent_in
    self.latent_dropout = latent_dropout
    if self.latent_dropout:
      self.lat_dp = tf.keras.layers.Dropout(0.2)
    self.dropout_prob = dropout_prob
    self.dropout = dropout
    self.xyz_in_all = xyz_in_all
    self.weight_norm = weight_norm

    for layer in range(0, self.num_layers - 1):
      if layer + 1 in latent_in:
        out_dim = dims[layer + 1] - dims[0]
      else:
        out_dim = dims[layer + 1]
        if self.xyz_in_all and layer != self.num_layers - 2:
          out_dim -= 3

      if weight_norm and layer in self.norm_layers:
        setattr(
            self,
            "lin" + str(layer),
            tfa.layers.WeightNormalization(tf.keras.Dense(
                out_dim, input_shape=(None, dims[layer])))
        )
      else:
        setattr(self, "lin" + str(layer),
                tf.keras.layers.Dense(out_dim, input_shape=(None, dims[layer])))

      if (
          (not weight_norm)
          and self.norm_layers is not None
          and layer in self.norm_layers
      ):
        setattr(self, "bn" + str(layer),
                tf.keras.layers.LayerNormalization(input_shape=(None, out_dim)))
      if self.dropout is not None and layer in self.dropout:
        setattr(self, "dp" + str(layer),
                tf.keras.layers.Dropout(self.dropout_prob))

    self.use_tanh = use_tanh
    if use_tanh:
      self.tanh = tf.keras.layers.Activation('tanh')

    self.relu = tf.keras.layers.Activation('relu')

    self.th = tf.keras.layers.Activation('tanh')

  # input: N x (L+3)

  def call(self, inp, training=False):
    xyz = inp[:, -3:]

    if inp.shape[1] > 3 and self.latent_dropout:
      latent_vecs = inp[:, :-3]
      latent_vecs = self.lat_dp(latent_vecs)
      x = tf.concat([latent_vecs, xyz], axis=1)
    else:
      x = inp

    for layer in range(0, self.num_layers - 1):
      lin = getattr(self, "lin" + str(layer))
      if layer in self.latent_in:
        x = tf.concat([x, inp], axis=1)
      elif layer != 0 and self.xyz_in_all:
        x = tf.concat([x, xyz], axis=1)
      x = lin(x)
      # last layer Tanh
      if layer == self.num_layers - 2 and self.use_tanh:
        x = self.tanh(x)
      if layer < self.num_layers - 2:
        if (
            self.norm_layers is not None
            and layer in self.norm_layers
            and not self.weight_norm
        ):
          bn = getattr(self, "bn" + str(layer))
          x = bn(x, training=training)
        x = self.relu(x)
        if self.dropout is not None and layer in self.dropout:
          dp = getattr(self, "dp" + str(layer))
          x = dp(x)

    if hasattr(self, "th"):
      x = self.th(x)

    return x
