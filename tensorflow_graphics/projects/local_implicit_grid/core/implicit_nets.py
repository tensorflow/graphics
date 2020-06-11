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
"""Implementations of various implicit function networks architectures.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

layers = tf.keras.layers


class ImNet(layers.Layer):
  """ImNet layer keras implementation.
  """

  def __init__(self, dim=3, in_features=128, out_features=1, num_filters=128,
               activation=tf.nn.leaky_relu, name='im_net'):
    """Initialization.

    Args:
      dim: int, dimension of input points.
      in_features: int, length of input features (i.e., latent code).
      out_features: number of output features.
      num_filters: int, width of the second to last layer.
      activation: tf activation op.
      name: str, name of the layer.
    """
    super(ImNet, self).__init__(name=name)
    self.dim = dim
    self.in_features = in_features
    self.dimz = dim + in_features
    self.out_features = out_features
    self.num_filters = num_filters
    self.activ = activation
    self.fc0 = layers.Dense(num_filters*16, name='dense_1')
    self.fc1 = layers.Dense(num_filters*8, name='dense_2')
    self.fc2 = layers.Dense(num_filters*4, name='dense_3')
    self.fc3 = layers.Dense(num_filters*2, name='dense_4')
    self.fc4 = layers.Dense(num_filters*1, name='dense_5')
    self.fc5 = layers.Dense(out_features, name='dense_6')
    self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

  def call(self, x, training=False):
    """Forward method.

    Args:
      x: `[batch_size, dim+in_features]` tensor, inputs to decode.
      training: bool, flag indicating training phase.
    Returns:
      x_: output through this layer.
    """
    x_ = x
    for dense in self.fc[:4]:
      x_ = self.activ(dense(x_))
      x_ = tf.concat([x_, x], axis=-1)
    x_ = self.activ(self.fc4(x_))
    x_ = self.fc5(x_)
    return x_
