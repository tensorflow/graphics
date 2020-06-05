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
"""Utility functions for keras layers."""

import tensorflow as tf

layers = tf.keras.layers
initializer = tf.keras.initializers.glorot_normal()
rate = 0.7


def norm_layer(tensor, normalization):
  if normalization.lower() == 'batchnorm':
    tensor = layers.BatchNormalization()(tensor)
  return tensor


def upconv(tensor, nfilters, size, strides,
           alpha_lrelu=0.2, normalization='None'):
  """Upconvolution as upsampling and convolution."""
  tensor = layers.UpSampling2D()(tensor)
  tensor = layers.Conv2D(nfilters, size,
                         strides=strides,
                         padding='same',
                         kernel_initializer=initializer,
                         use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)
  tensor = layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if normalization.lower() == 'dropout':
    tensor = layers.Dropout(rate)(tensor)
  return tensor


def conv_block_3d(tensor, nfilters, size, strides,
                  alpha_lrelu=0.2, normalization='None', relu=True):
  """3D convolution block with normalization and leaky relu."""
  tensor = layers.Conv3D(nfilters, size,
                         strides=strides,
                         padding='same',
                         kernel_initializer=initializer,
                         use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)

  if relu:
    tensor = layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if normalization.lower() == 'dropout':
    tensor = layers.Dropout(rate)(tensor)
  return tensor


def conv_t_block_3d(tensor, nfilters, size, strides,
                    alpha_lrelu=0.2, normalization='None', relu=True):
  """2D transpose convolution block with normalization and leaky relu."""
  tensor = layers.Conv3DTranspose(nfilters, size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)
  if relu:
    tensor = layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if normalization.lower() == 'dropout':
    tensor = layers.Dropout(rate)(tensor)
  return tensor


def conv_block_2d(tensor, nfilters, size, strides,
                  alpha_lrelu=0.2, normalization='None'):
  """2D convolution block with normalization and leaky relu."""
  tensor = layers.Conv2D(nfilters, size,
                         strides=strides,
                         padding='same',
                         kernel_initializer=initializer,
                         use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)

  tensor = layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if normalization.lower() == 'dropout':
    tensor = layers.Dropout(rate)(tensor)
  return tensor


def conv_t_block_2d(tensor, nfilters, size, strides,
                    alpha_lrelu=0.2, normalization='None'):
  """2D transpose convolution block with normalization and leaky relu."""
  tensor = layers.Conv2DTranspose(nfilters, size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False)(tensor)

  tensor = norm_layer(tensor, normalization)

  tensor = layers.LeakyReLU(alpha=alpha_lrelu)(tensor)
  if normalization.lower() == 'dropout':
    tensor = layers.Dropout(rate)(tensor)
  return tensor


def residual_block_2d(x, nfilters, strides=(1, 1), normalization='None'):
  """2D residual block."""
  shortcut = x

  x = layers.Conv2D(nfilters,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding='same',
                    kernel_initializer=initializer)(x)
  x = norm_layer(x, normalization)
  x = layers.LeakyReLU()(x)

  x = layers.Conv2D(nfilters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=initializer)(x)
  x = norm_layer(x, normalization)

  if strides != (1, 1):
    shortcut = layers.Conv2D(nfilters,
                             kernel_size=(1, 1),
                             strides=strides,
                             padding='same')(shortcut)
    x = norm_layer(x, normalization)

  x = layers.add([shortcut, x])
  x = layers.LeakyReLU()(x)

  return x


def residual_block_3d(x, nfilters, strides=(1, 1, 1), normalization='None'):
  """3D residual block."""
  shortcut = x

  x = layers.Conv3D(nfilters,
                    kernel_size=(3, 3, 3),
                    strides=strides,
                    padding='same',
                    kernel_initializer=initializer)(x)
  x = norm_layer(x, normalization)
  x = layers.LeakyReLU()(x)

  x = layers.Conv3D(nfilters,
                    kernel_size=(3, 3, 3),
                    strides=(1, 1, 1),
                    padding='same',
                    kernel_initializer=initializer)(x)
  x = norm_layer(x, normalization)

  if strides != (1, 1, 1):
    shortcut = layers.Conv3D(nfilters,
                             kernel_size=(1, 1, 1),
                             strides=strides,
                             padding='same')(shortcut)
    x = norm_layer(x, normalization)

  x = layers.add([shortcut, x])
  x = layers.LeakyReLU()(x)

  return x
