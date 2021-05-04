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
"""Network layer utilities."""
import tensorflow as tf

layers = tf.keras.layers
initializer = tf.keras.initializers.glorot_normal()


@tf.function
def process_voxels(voxels, kernel_size=3):
  kernel = [kernel_size, kernel_size, kernel_size]
  voxels = tf.nn.pool(voxels, kernel, 'MAX', [1, 1, 1], padding='SAME')
  voxels = tf.nn.pool(voxels, kernel, 'AVG', [1, 1, 1], padding='SAME')
  return voxels


def norm_layer(tensor, normalization):
  if normalization.lower() == 'batchnorm':
    tensor = layers.BatchNormalization()(tensor)
  return tensor


def conv_block_3d(tensor, nfilters, size, strides,
                  alpha_lrelu=0.2, normalization='None', relu=True, rate=0.7):
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
                    alpha_lrelu=0.2, normalization='None', relu=True, rate=0.7):
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

