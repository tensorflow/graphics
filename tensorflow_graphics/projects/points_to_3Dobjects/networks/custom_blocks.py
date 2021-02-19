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
# python3
"""Custom blocks."""
import tensorflow as tf


class IdentityLayer(tf.keras.layers.Layer):

  def __init__(self, name='identity'):
    super(IdentityLayer, self).__init__(name=name)

  def call(self, inputs, **kwargs):
    return inputs


class SkipConvolution(tf.keras.layers.Layer):
  """Convolution block with skip connections."""

  def __init__(self,
               out_dim,
               stride,
               norm_type,
               kernel_regularizer,
               name='SkipConvolution'):
    super(SkipConvolution, self).__init__(name=name)
    self.conv = tf.keras.layers.Conv2D(
        out_dim,
        1,
        kernel_regularizer=kernel_regularizer(),
        use_bias=False,
        strides=stride,
        name='conv')
    self.norm = norm_type()

  def call(self, inputs, **kwargs):
    x = self.conv(inputs)
    x = self.norm(x)
    return x


class ConvolutionalBlock(tf.keras.layers.Layer):
  """Block that aggregates Convolution + Norm layer + ReLU.

    Attributes:
      kernel_size: Integer, convolution kernel size.
      out_dim: Integer, number of filter in the convolution.
      norm_type: Object type of the norm to use.
      stride: Integer, stride used in the convolution.
      input_shape: Tuple with the input shape i.e. (None, 512, 512, 3).
      padding: String, padding type used in the convolution.
      relu: Boolean, use relu at the end.
      name: String, name of the block.
  """

  def __init__(self,
               kernel_size,
               out_dim,
               norm_type,
               kernel_regularizer,
               stride=1,
               input_shape=None,
               padding='same',
               relu=True,
               name='ConvolutionalBlock'):
    super(ConvolutionalBlock, self).__init__(name=name)
    if kernel_size > 1:
      # Padding corresponding to kernel and stride (kernel, stride): pad
      pad_vals = {7: (3, 3), 3: (1, 1)}
      padding = 'valid'
      self.pad = tf.keras.layers.ZeroPadding2D(pad_vals[kernel_size])
    else:
      self.pad = IdentityLayer(name='identity')
    if input_shape is not None:
      self.conv = tf.keras.layers.Conv2D(
          out_dim,
          kernel_size,
          kernel_regularizer=kernel_regularizer(),
          use_bias=False,
          strides=stride,
          padding=padding,
          input_shape=input_shape,
          name='conv')
    else:
      self.conv = tf.keras.layers.Conv2D(
          out_dim,
          kernel_size,
          kernel_regularizer=kernel_regularizer(),
          use_bias=False,
          strides=stride,
          padding=padding,
          name='conv')
    self.norm = norm_type()
    if relu:
      self.relu = tf.keras.layers.ReLU(name='relu')
    else:
      self.relu = IdentityLayer(name='identity')

  def call(self, inputs, **kwargs):
    x = self.pad(inputs)
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x


class ResidualBlock(tf.keras.layers.Layer):
  """Block that aggregations the operations in a residual block.

  Attributes:
    out_dim: Integer, number of filter in the convolution.
    norm_type: Object type of the norm to use.
    skip: Boolean, apply convolution in the skip connection.
    kernel_size: Integer, convolution kernel size.
    stride: Integer, stride used in the convolution.
    padding: String, padding type used in the convolution.
    name: String, name of the block.
  """

  def __init__(self,
               out_dim,
               norm_type,
               kernel_regularizer,
               skip=False,
               kernel_size=3,
               stride=1,
               padding='same',
               name='ResidualBlock'):
    super(ResidualBlock, self).__init__(name=name)
    self.conv_block = ConvolutionalBlock(kernel_size, out_dim, norm_type,
                                         kernel_regularizer, stride)

    self.conv = tf.keras.layers.Conv2D(
        out_dim,
        kernel_size,
        kernel_regularizer=kernel_regularizer(),
        use_bias=False,
        strides=1,
        padding=padding,
        name='conv')
    self.norm = norm_type()

    if skip:
      self.skip = SkipConvolution(out_dim, stride, norm_type,
                                  kernel_regularizer, 'skip')
    else:
      self.skip = IdentityLayer('identity')

    self.relu = tf.keras.layers.ReLU(name='relu')

  def call(self, inputs, **kwargs):
    x = self.conv_block(inputs)
    x = self.conv(x)
    x = self.norm(x)
    x_skip = self.skip(inputs)
    x = self.relu(x + x_skip)
    return x
