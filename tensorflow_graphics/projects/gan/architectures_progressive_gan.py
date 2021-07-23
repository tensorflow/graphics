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
"""Network architectures from the progressive GAN paper.

Implemented according to the paper "Progressive growing of GANs for Improved
Quality, Stability, and Variation"
https://arxiv.org/abs/1710.10196

Intermediate outputs and inputs are supported for implementation of "MSG-GAN:
Multi-Scale Gradient GAN for Stable Image Synthesis"
https://arxiv.org/abs/1903.06048

The implementations are done using Keras models with the Functional API. Only a
subset of the architectures presented in the papers are implemented and
particularly progressive growing is not supported.
"""

import math
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf
import tensorflow_addons.layers.normalizations as tfa_normalizations

from tensorflow_graphics.projects.gan import keras_layers

_InitializerCallable = Callable[[tf.Tensor, tf.dtypes.DType], tf.Tensor]
_KerasInitializer = Union[_InitializerCallable, str]


def to_rgb(input_tensor: tf.Tensor,
           kernel_initializer: _KerasInitializer,
           name: Optional[str] = None) -> tf.Tensor:
  """Converts a feature map to an rgb output.

  Args:
    input_tensor: The input feature map.
    kernel_initializer: The kernel initializer to use.
    name: The name of the layer.

  Returns:
    The rgb image.
  """
  return keras_layers.FanInScaledConv2D(
      multiplier=1.0,
      filters=3,
      kernel_size=1,
      strides=1,
      kernel_initializer=kernel_initializer,
      padding='same',
      name=name)(
          input_tensor)


def create_generator(latent_code_dimension: int = 128,
                     upsampling_blocks_num_channels: Sequence[int] = (512, 256,
                                                                      128, 64),
                     relu_leakiness: float = 0.2,
                     kernel_initializer: Optional[_KerasInitializer] = None,
                     use_pixel_normalization: bool = True,
                     use_batch_normalization: bool = False,
                     generate_intermediate_outputs: bool = False,
                     normalize_latent_code: bool = True,
                     name: str = 'progressive_gan_generator') -> tf.keras.Model:
  """Creates a Keras model for the generator network architecture.

  This architecture is implemented according to the paper "Progressive growing
  of GANs for Improved Quality, Stability, and Variation"
  https://arxiv.org/abs/1710.10196
  The intermediate outputs are optionally provided for the architecture of
  "MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis"
  https://arxiv.org/abs/1903.06048

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    upsampling_blocks_num_channels: The number of channels for each upsampling
      block. This argument also determines how many upsampling blocks are added.
    relu_leakiness: Slope of the negative part of the leaky relu.
    kernel_initializer: Initializer of the kernel. If none TruncatedNormal is
      used.
    use_pixel_normalization: If pixel normalization layers should be inserted to
      the network.
    use_batch_normalization: If batch normalization layers should be inserted to
      the network.
    generate_intermediate_outputs: If true the model outputs a list of
      tf.Tensors with increasing resolution starting with the starting_size up
      to the final resolution output.
    normalize_latent_code: If true the latent code is normalized to unit length
      before feeding it to the network.
    name: The name of the Keras model.

  Returns:
     The created generator keras model object.
  """
  if kernel_initializer is None:
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=1.0)

  input_tensor = tf.keras.Input(shape=(latent_code_dimension,))
  if normalize_latent_code:
    maybe_normzlized_input_tensor = keras_layers.PixelNormalization(axis=1)(
        input_tensor)
  else:
    maybe_normzlized_input_tensor = input_tensor

  tensor = keras_layers.FanInScaledDense(
      multiplier=math.sqrt(2.0) / 4.0,
      units=4 * 4 * latent_code_dimension,
      kernel_initializer=kernel_initializer)(
          maybe_normzlized_input_tensor)
  tensor = tf.keras.layers.Reshape(target_shape=(4, 4, latent_code_dimension))(
      tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  if use_batch_normalization:
    tensor = tf.keras.layers.BatchNormalization()(tensor)
  if use_pixel_normalization:
    tensor = keras_layers.PixelNormalization(axis=3)(tensor)
  tensor = keras_layers.FanInScaledConv2D(
      filters=upsampling_blocks_num_channels[0],
      kernel_size=3,
      strides=1,
      padding='same',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  if use_batch_normalization:
    tensor = tf.keras.layers.BatchNormalization()(tensor)
  if use_pixel_normalization:
    tensor = keras_layers.PixelNormalization(axis=3)(tensor)

  outputs = []
  for index, channels in enumerate(upsampling_blocks_num_channels):
    if generate_intermediate_outputs:
      outputs.append(
          to_rgb(
              input_tensor=tensor,
              kernel_initializer=kernel_initializer,
              name='side_output_%d_conv' % index))
    tensor = keras_layers.TwoByTwoNearestNeighborUpSampling()(tensor)

    for _ in range(2):
      tensor = keras_layers.FanInScaledConv2D(
          filters=channels,
          kernel_size=3,
          strides=1,
          padding='same',
          kernel_initializer=kernel_initializer)(
              tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
      if use_batch_normalization:
        tensor = tf.keras.layers.BatchNormalization()(tensor)
      if use_pixel_normalization:
        tensor = keras_layers.PixelNormalization(axis=3)(tensor)

  tensor = to_rgb(
      input_tensor=tensor,
      kernel_initializer=kernel_initializer,
      name='final_output')
  if generate_intermediate_outputs:
    outputs.append(tensor)

    return tf.keras.Model(inputs=input_tensor, outputs=outputs, name=name)
  else:
    return tf.keras.Model(inputs=input_tensor, outputs=tensor, name=name)


def create_conv_layer(use_fan_in_scaled_kernel: bool = False,
                      multiplier: float = math.sqrt(2),
                      **kwargs) -> tf.keras.layers.Conv2D:
  """Creates a convolutional layer.

  Args:
    use_fan_in_scaled_kernel: Whether to use a FanInScaledConv2D or a standard
      Conv2D layer.
    multiplier: Additional multiplier used only for FanInSclaedConv2D layer.
    **kwargs: Keyword arguments forwarded to the convolutional layers.

  Returns:
    The created convolutional layer instance.
  """
  if use_fan_in_scaled_kernel:
    return keras_layers.FanInScaledConv2D(multiplier=multiplier, **kwargs)
  else:
    return tf.keras.layers.Conv2D(**kwargs)


def from_rgb(input_tensor: tf.Tensor,
             use_fan_in_scaled_kernel: bool,
             num_channels: int,
             kernel_initializer: _KerasInitializer,
             relu_leakiness: float,
             name: str = 'from_rgb') -> tf.Tensor:
  """Converts a rgb input to a feature map.

  Args:
    input_tensor: The input feature map.
    use_fan_in_scaled_kernel: If a fan in scaled kernel should be used.
    num_channels: The number of output channels.
    kernel_initializer: The kernel initializer to use.
    relu_leakiness: The leakiness of the ReLU.
    name: The name of the block.

  Returns:
    The feature map.
  """
  with tf.name_scope(name):
    output = create_conv_layer(
        use_fan_in_scaled_kernel=use_fan_in_scaled_kernel,
        filters=num_channels,
        kernel_size=1,
        strides=1,
        kernel_initializer=kernel_initializer,
        padding='same')(
            input_tensor)
    return tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(output)


def create_discriminator(
    downsampling_blocks_num_channels: Sequence[Sequence[int]] = ((64, 128),
                                                                 (128, 128),
                                                                 (256, 256),
                                                                 (512, 512)),
    relu_leakiness: float = 0.2,
    kernel_initializer: Optional[_KerasInitializer] = None,
    use_fan_in_scaled_kernels: bool = True,
    use_layer_normalization: bool = False,
    use_intermediate_inputs: bool = False,
    use_antialiased_bilinear_downsampling: bool = False,
    name: str = 'progressive_gan_discriminator'):
  """Creates a Keras model for the discriminator architecture.

  This architecture is implemented according to the paper "Progressive growing
  of GANs for Improved Quality, Stability, and Variation"
  https://arxiv.org/abs/1710.10196
  The intermediate outputs can optionally be given as input for the architecture
  of "MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis"
  https://arxiv.org/abs/1903.06048

  Args:
    downsampling_blocks_num_channels: The number of channels in the downsampling
      blocks for each block the number of channels for the first and second
      convolution are specified.
    relu_leakiness: Slope of the negative part of the leaky relu.
    kernel_initializer: Initializer of the kernel. If none TruncatedNormal is
      used.
    use_fan_in_scaled_kernels: This rescales the kernels using the scale factor
      from the he initializer, which implements the equalized learning rate.
    use_layer_normalization: If layer normalization layers should be inserted to
      the network.
    use_intermediate_inputs: If true the model expects a list of tf.Tensors with
      increasing resolution starting with the starting_size up to the final
      resolution as input.
    use_antialiased_bilinear_downsampling: If true the downsampling operation is
      ani-aliased bilinear downsampling with a [1, 3, 3, 1] tent kernel. If
      false standard bilinear downsampling, i.e. average pooling is used ([1, 1]
      tent kernel).
    name: The name of the Keras model.

  Returns:
    The generated discriminator keras model.
  """
  if kernel_initializer is None:
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=1.0)

  if use_intermediate_inputs:
    inputs = tuple(
        tf.keras.Input(shape=(None, None, 3))
        for _ in range(len(downsampling_blocks_num_channels) + 1))
    tensor = inputs[-1]
  else:
    input_tensor = tf.keras.Input(shape=(None, None, 3))
    tensor = input_tensor

  tensor = from_rgb(
      tensor,
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      num_channels=downsampling_blocks_num_channels[0][0],
      kernel_initializer=kernel_initializer,
      relu_leakiness=relu_leakiness)
  if use_layer_normalization:
    tensor = tfa_normalizations.GroupNormalization(groups=1)(tensor)

  for index, (channels_1,
              channels_2) in enumerate(downsampling_blocks_num_channels):
    tensor = create_conv_layer(
        use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
        filters=channels_1,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer)(
            tensor)
    tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
    if use_layer_normalization:
      tensor = tfa_normalizations.GroupNormalization(groups=1)(tensor)
    tensor = create_conv_layer(
        use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
        filters=channels_2,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer)(
            tensor)
    tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
    if use_layer_normalization:
      tensor = tfa_normalizations.GroupNormalization(groups=1)(tensor)
    if use_antialiased_bilinear_downsampling:
      tensor = keras_layers.Blur2D()(tensor)
    tensor = tf.keras.layers.AveragePooling2D()(tensor)

    if use_intermediate_inputs:
      tensor = tf.keras.layers.Concatenate()([inputs[-index - 2], tensor])

  tensor = create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      filters=downsampling_blocks_num_channels[-1][1],
      kernel_size=3,
      strides=1,
      padding='same',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  if use_layer_normalization:
    tensor = tfa_normalizations.GroupNormalization(groups=1)(tensor)

  tensor = create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      filters=downsampling_blocks_num_channels[-1][1],
      kernel_size=4,
      strides=1,
      padding='valid',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  if use_layer_normalization:
    tensor = tfa_normalizations.GroupNormalization(groups=1)(tensor)

  tensor = create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      multiplier=1.0,
      filters=1,
      kernel_size=1,
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.Reshape((-1,))(tensor)

  if use_intermediate_inputs:
    return tf.keras.Model(inputs=inputs, outputs=tensor, name=name)
  else:
    return tf.keras.Model(inputs=input_tensor, outputs=tensor, name=name)
