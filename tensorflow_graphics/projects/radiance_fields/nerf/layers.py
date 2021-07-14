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
"""Appearance network layer utilities."""
import tensorflow as tf

layers = tf.keras.layers


def dense_block(tensor, n_filters, n_layers=2, activation=layers.ReLU(),
                seed=None):
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)
  for _ in range(n_layers):
    tensor = layers.Dense(n_filters,
                          activation=activation,
                          kernel_initializer=initializer)(tensor)
  return tensor


def concat_block(tensor, n_filters, n_layers=2, activation=layers.ReLU(),
                 seed=None):
  shortcut = tensor
  tensor = dense_block(tensor, n_filters, n_layers, activation, seed)
  return layers.concatenate([shortcut, tensor], -1)
