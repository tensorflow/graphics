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
"""Geometry network layer utilities."""
import tensorflow as tf


def norm_layer(tensor, normalization):
  if normalization and normalization.lower() == 'batchnorm':
    tensor = tf.keras.layers.BatchNormalization()(tensor)
  return tensor


def conv_block_3d(tensor, num_filters, size, strides,
                  normalization=None, dropout=False,
                  alpha_lrelu=0.2, relu=True, rate=0.7):
  """3D convolution block with normalization and leaky relu."""
  tensor = tf.keras.layers.Conv3D(
      num_filters,
      size,
      strides=strides,
      padding='same',
      kernel_initializer=tf.keras.initializers.glorot_normal(),
      use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)

  if relu:
    tensor = tf.keras.layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if dropout:
    tensor = tf.keras.layers.Dropout(rate)(tensor)
  return tensor


def conv_t_block_3d(tensor, num_filters, size, strides,
                    normalization=None, dropout=False,
                    alpha_lrelu=0.2, relu=True, rate=0.7):
  """2D transpose convolution block with normalization and leaky relu."""
  tensor = tf.keras.layers.Conv3DTranspose(
      num_filters,
      size,
      strides=strides,
      padding='same',
      kernel_initializer=tf.keras.initializers.glorot_normal(),
      use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)
  if relu:
    tensor = tf.keras.layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if dropout:
    tensor = tf.keras.layers.Dropout(rate)(tensor)
  return tensor

