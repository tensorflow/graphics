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
"""Network architectures from the StyleGan v2 paper.

Implements architectures according to the paper "Analyzing and Improving the
Image Quality of StyleGAN"
https://arxiv.org/pdf/1912.04958.pdf

The implementations are done using Keras models with the Functional API. Only a
subset of the architectures presented in the papers are implemented.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow_graphics.projects.gan import architectures_progressive_gan
from tensorflow_graphics.projects.gan import architectures_style_gan
from tensorflow_graphics.projects.gan import keras_layers

_InitializerCallable = Callable[[tf.Tensor, tf.dtypes.DType], tf.Tensor]
_KerasInitializer = Union[_InitializerCallable, str]

CUSTOM_LAYERS = keras_layers.CUSTOM_LAYERS


def _maybe_upsample_and_add_outputs(
    current_level_output,
    previous_level_output: Optional[tf.Tensor] = None,
    use_bilinear_upsampling: bool = True):
  """Upsamples and adds previous and current level or returns current level.

  Args:
    current_level_output: Output of the current level.
    previous_level_output: Output of the previous level. If None
      current_level_output will be returned.
    use_bilinear_upsampling: If true bilinear upsampling is used else nearest
      neighbor.

  Returns:
    The sum of the upsampled previous level and the current level.
  """
  if previous_level_output is None:
    return current_level_output
  else:
    upsampled_output = keras_layers.TwoByTwoNearestNeighborUpSampling()(
        previous_level_output)
    if use_bilinear_upsampling:
      upsampled_output = keras_layers.Blur2D()(upsampled_output)
    return current_level_output + upsampled_output


def get_noise_dimensions(
    num_upsampling_blocks: int) -> Sequence[Tuple[int, int, int]]:
  """Returns the dimensions of the noise inputs for noise input.

  This function can be used to determine the shape of the noise that needs to be
  fed into the synthesis network or generator network whenn use_noise_inputs is
  enabled. The dimensions are given as [height, width, channels].

  Args:
    num_upsampling_blocks: The number of upsampling blocks in the synthesis
      network.

  Returns:
    The dimensions of the noise inputs.
  """
  noise_dimensions = [(4, 4, 1)]
  size = 8
  for _ in range(num_upsampling_blocks):
    for _ in range(2):
      noise_dimensions.append((size, size, 1))
    size *= 2
  return noise_dimensions


def _create_noise_inputs(num_upsampling_blocks: int) -> List[tf.Tensor]:
  """Creates the noise input layers."""
  noise_dimensions = get_noise_dimensions(num_upsampling_blocks)
  noise_inputs = [
      tf.keras.Input(noise_dimension) for noise_dimension in noise_dimensions
  ]
  return noise_inputs


def create_synthesis_network(
    latent_code_dimension: int = 128,
    upsampling_blocks_num_channels: Sequence[int] = (512, 256, 128, 64),
    relu_leakiness: float = 0.2,
    use_bilinear_upsampling: bool = True,
    use_noise_inputs: bool = False,
    name: str = 'synthesis'):
  """Creates the synthesis network using the functional API.

  The function creates the synthesis network as defined in "Analyzing and
  Improving the Image Quality of StyleGAN" https://arxiv.org/pdf/1912.04958.pdf
  using the Keras functional API.

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    upsampling_blocks_num_channels: The number of channels for each upsampling
      block. This argument also determines how many upsampling blocks are added.
    relu_leakiness: Slope of the negative part of the leaky relu.
    use_bilinear_upsampling: If true bilinear upsampling is used.
    use_noise_inputs: If the model takes noise as input, if false noise is
      sampled randomly.
    name: The name of the Keras model.

  Returns:
    The synthesis network.
  """
  kernel_initializer = tf.keras.initializers.TruncatedNormal(
      mean=0.0, stddev=1.0)
  mapped_latent_code_input = tf.keras.Input(shape=(latent_code_dimension,))

  if use_noise_inputs:
    noise_inputs = _create_noise_inputs(len(upsampling_blocks_num_channels))

  tensor = keras_layers.LearnedConstant()(mapped_latent_code_input)
  tensor = keras_layers.DemodulatedConvolution(
      filters=upsampling_blocks_num_channels[0],
      kernel_size=3)((tensor, mapped_latent_code_input))
  if use_noise_inputs:
    tensor = keras_layers.Noise()((tensor, noise_inputs[0]))
  else:
    tensor = keras_layers.Noise()(tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)

  output = None
  for index, channels in enumerate(upsampling_blocks_num_channels):
    output = _maybe_upsample_and_add_outputs(
        architectures_progressive_gan.to_rgb(
            input_tensor=tensor,
            kernel_initializer=kernel_initializer,
            name='side_output_%d_conv' % index),
        output,
        use_bilinear_upsampling=use_bilinear_upsampling)
    tensor = keras_layers.TwoByTwoNearestNeighborUpSampling()(tensor)
    if use_bilinear_upsampling:
      tensor = keras_layers.Blur2D()(tensor)
    for inner_index in range(2):
      tensor = keras_layers.DemodulatedConvolution(
          filters=channels, kernel_size=3)((tensor, mapped_latent_code_input))
      if use_noise_inputs:
        noise_index = 2 * index + inner_index + 1
        tensor = keras_layers.Noise()((tensor, noise_inputs[noise_index]))
      else:
        tensor = keras_layers.Noise()(tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)

  output = _maybe_upsample_and_add_outputs(
      architectures_progressive_gan.to_rgb(
          input_tensor=tensor, kernel_initializer=kernel_initializer),
      output,
      use_bilinear_upsampling=use_bilinear_upsampling)

  if use_noise_inputs:
    inputs = [mapped_latent_code_input] + noise_inputs
  else:
    inputs = mapped_latent_code_input

  return tf.keras.Model(inputs=inputs, outputs=output, name=name)


def create_style_based_generator(
    latent_code_dimension: int = 128,
    upsampling_blocks_num_channels: Sequence[int] = (512, 256, 128, 64),
    relu_leakiness: float = 0.2,
    use_bilinear_upsampling: bool = True,
    normalize_latent_code: bool = True,
    use_noise_inputs: bool = False,
    name: str = 'style_gan_v2_generator'
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
  """Creates a Keras model for the style based generator network architecture.

  This architecture is implemented accodring to "Analyzing and Improving the
  Image Quality of StyleGAN"
  https://arxiv.org/pdf/1912.04958.pdf

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    upsampling_blocks_num_channels: The number of channels for each upsampling
      block. This argument also determines how many upsampling blocks are added.
    relu_leakiness: Slope of the negative part of the leaky relu.
    use_bilinear_upsampling: If true bilinear upsampling is used.
    normalize_latent_code: If true the latent code is normalized to unit length
      before feeding it to the network.
    use_noise_inputs: If the model takes noise as input, if false noise is
      sampled randomly.
    name: The name of the Keras model.

  Returns:
     Three Keras models. The whole generator, only the mapping network and only
     the synthesis network.
  """
  mapping_network = architectures_style_gan.create_mapping_network(
      latent_code_dimension=latent_code_dimension,
      normalize_latent_code=normalize_latent_code,
      relu_leakiness=relu_leakiness)
  synthesis_network = create_synthesis_network(
      latent_code_dimension=latent_code_dimension,
      upsampling_blocks_num_channels=upsampling_blocks_num_channels,
      relu_leakiness=relu_leakiness,
      use_bilinear_upsampling=use_bilinear_upsampling,
      use_noise_inputs=use_noise_inputs)

  input_tensor = tf.keras.Input(shape=(latent_code_dimension,))
  mapped_latent_code = mapping_network(input_tensor)

  if use_noise_inputs:
    noise_inputs = _create_noise_inputs(len(upsampling_blocks_num_channels))
    synthesis_inputs = [mapped_latent_code] + noise_inputs
    generator_inputs = [input_tensor] + noise_inputs
  else:
    synthesis_inputs = mapped_latent_code
    generator_inputs = input_tensor

  generated_images = synthesis_network(synthesis_inputs)
  generator = tf.keras.Model(
      inputs=generator_inputs, outputs=generated_images, name=name)

  return generator, mapping_network, synthesis_network


def create_discriminator(
    downsampling_blocks_num_channels: Sequence[Sequence[int]] = ((64, 128),
                                                                 (128, 128),
                                                                 (256, 256),
                                                                 (512, 512)),
    relu_leakiness: float = 0.2,
    kernel_initializer: Optional[_KerasInitializer] = None,
    use_fan_in_scaled_kernels: bool = True,
    use_antialiased_bilinear_downsampling: bool = False,
    num_channels: int = 3,
    name: str = 'style_gan_v2_discriminator'):
  """Creates a Keras model for the discriminator architecture.

  This architecture is implemented accodring to "Analyzing and Improving the
  Image Quality of StyleGAN"
  https://arxiv.org/pdf/1912.04958.pdf

  Args:
    downsampling_blocks_num_channels: The number of channels in the downsampling
      blocks for each block the number of channels for the first and second
      convolution are specified.
    relu_leakiness: Slope of the negative part of the leaky relu.
    kernel_initializer: Initializer of the kernel. If none TruncatedNormal is
      used.
    use_fan_in_scaled_kernels: This rescales the kernels using the scale factor
      from the he initializer, which implements the equalized learning rate.
    use_antialiased_bilinear_downsampling: If true the downsampling operation is
      ani-aliased bilinear downsampling with a [1, 3, 3, 1] tent kernel. If
      false standard bilinear downsampling, i.e. average pooling is used ([1, 1]
      tent kernel).
    num_channels: The number of channels of the input tensor.
    name: The name of the Keras model.

  Returns:
    The generated discriminator keras model.
  """
  if kernel_initializer is None:
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=1.0)

  input_tensor = tf.keras.Input(shape=(None, None, num_channels))
  tensor = architectures_progressive_gan.from_rgb(
      input_tensor=input_tensor,
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      num_channels=downsampling_blocks_num_channels[0][0],
      kernel_initializer=kernel_initializer,
      relu_leakiness=relu_leakiness)

  for index, (channels_1,
              channels_2) in enumerate(downsampling_blocks_num_channels):
    with tf.name_scope(f'downsampling_block_{index}'):
      shortcut = tensor

      shortcut = architectures_progressive_gan.create_conv_layer(
          use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
          filters=channels_2,
          kernel_size=1,
          strides=1,
          use_bias=False,
          kernel_initializer=kernel_initializer)(
              shortcut)
      shortcut = tf.keras.layers.AveragePooling2D()(shortcut)
      if use_antialiased_bilinear_downsampling:
        shortcut = keras_layers.Blur2D()(shortcut)

      tensor = architectures_progressive_gan.create_conv_layer(
          use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
          filters=channels_1,
          kernel_size=3,
          strides=1,
          padding='same',
          kernel_initializer=kernel_initializer)(
              tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
      tensor = architectures_progressive_gan.create_conv_layer(
          use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
          filters=channels_2,
          kernel_size=3,
          strides=1,
          padding='same',
          kernel_initializer=kernel_initializer)(
              tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
      if use_antialiased_bilinear_downsampling:
        tensor = keras_layers.Blur2D()(tensor)
      tensor = tf.keras.layers.AveragePooling2D()(tensor)

      # Adding residual connection and normalizing the variance.
      tensor = (tensor + shortcut) * 1 / math.sqrt(2.0)

  tensor = architectures_progressive_gan.create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      filters=downsampling_blocks_num_channels[-1][1],
      kernel_size=3,
      strides=1,
      padding='same',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)

  tensor = architectures_progressive_gan.create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      filters=downsampling_blocks_num_channels[-1][1],
      kernel_size=4,
      strides=1,
      padding='valid',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)

  tensor = architectures_progressive_gan.create_conv_layer(
      use_fan_in_scaled_kernel=use_fan_in_scaled_kernels,
      multiplier=1.0,
      filters=1,
      kernel_size=1,
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = tf.keras.layers.Reshape((-1,))(tensor)

  return tf.keras.Model(inputs=input_tensor, outputs=tensor, name=name)
