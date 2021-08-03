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
"""Network architectures from the style GAN paper.

Implements the style based generator of "A Style-Based Generator Architecture
for Generative Adversarial Networks".
https://arxiv.org/pdf/1812.04948.pdf"

Intermediate outputs and inputs are supported for implementation of "MSG-GAN:
Multi-Scale Gradient GAN for Stable Image Synthesis"
https://arxiv.org/abs/1903.06048

The implementations are done using Keras models with the Functional API. Only a
subset of the architectures presented in the papers are implemented and
particularly progressive growing is not supported.
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_addons.layers.normalizations as tfa_normalizations
from tensorflow_graphics.projects.gan import architectures_progressive_gan
from tensorflow_graphics.projects.gan import keras_layers

_InitializerCallable = Callable[[tf.Tensor, tf.dtypes.DType], tf.Tensor]
_KerasInitializer = Union[_InitializerCallable, str]

CUSTOM_LAYERS = keras_layers.CUSTOM_LAYERS


def apply_style_with_adain(mapped_latent_code: tf.Tensor,
                           input_tensor: tf.Tensor) -> tf.Tensor:
  """Applies a style using Adaptive Instance Normalization (AdaIN).

  This function first converts the mapped latent code to a scale and offset for
  AdaIN using an affine transormation, then the style is applied using AdaIN.

  Args:
    mapped_latent_code: The mapped latent code that contains the style.
    input_tensor: The input tensor to which the style should be applied.

  Returns:
    The input_tensor after applying the style.
  """
  number_of_channels = input_tensor.shape[-1]
  scale_and_offset = keras_layers.FanInScaledDense(
      multiplier=1.0,
      units=2 * number_of_channels,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(
          mean=0.0, stddev=1.0))(
              mapped_latent_code)

  normalized_input_tensor = tfa_normalizations.InstanceNormalization(
      center=False, scale=False)(
          input_tensor)
  scale = scale_and_offset[:, :number_of_channels][:, tf.newaxis, tf.newaxis, :]
  offset = scale_and_offset[:, number_of_channels:][:, tf.newaxis,
                                                    tf.newaxis, :]
  # Adding 1.0 to the scale to achieve initialization of the bias to 1.0.
  return tf.keras.layers.add((tf.keras.layers.multiply(
      (normalized_input_tensor, scale + 1.0)), offset))


def create_mapping_network(latent_code_dimension: int = 128,
                           output_dimension: Optional[int] = None,
                           normalize_latent_code: bool = True,
                           relu_leakiness: float = 0.2,
                           num_layers: int = 8,
                           learning_rate_multiplier: float = 0.01,
                           name: str = 'mapping') -> tf.keras.Model:
  """Creates the mapping network using the functional API.

  The function creates the mapping network as defined in "A Style-Based
  Generator Architecture for Generative Adversarial Networks"
  https://arxiv.org/abs/1812.04948 using the Keras functional API.

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    output_dimension: The dimension of the output. By default it is latent code
      dimension.
    normalize_latent_code: If true the latent code is normalized to unit length
      before feeding it to the network.
    relu_leakiness: The leakiness of the relu.
    num_layers: The number of dense layers that will be used.
    learning_rate_multiplier: The learning rate multiplier that is used in the
      scaled dense layer. This assumes that the network is trained using ADAM or
      a similar optimizer that normalizes the length of the gradient.
    name: The name of the Keras model.

  Returns:
    The mapping network.
  """
  if output_dimension is None:
    output_dimension = latent_code_dimension

  input_tensor = tf.keras.Input(shape=(latent_code_dimension,))
  if normalize_latent_code:
    maybe_normalized_input_tensor = keras_layers.PixelNormalization(axis=1)(
        input_tensor)
  else:
    maybe_normalized_input_tensor = input_tensor

  tensor = maybe_normalized_input_tensor
  for i in range(num_layers):
    with tf.name_scope(name='mapping_layer_%d' % i):
      # The kernel_multiplier implements the reduced learning rate of the
      # mapping network.
      tensor = keras_layers.FanInScaledDense(
          units=output_dimension,
          kernel_multiplier=learning_rate_multiplier,
          bias_multiplier=learning_rate_multiplier,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              mean=0.0, stddev=1.0 / learning_rate_multiplier))(
                  tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  return tf.keras.Model(inputs=input_tensor, outputs=tensor, name=name)


def create_synthesis_network(latent_code_dimension: int = 128,
                             upsampling_blocks_num_channels: Sequence[int] = (
                                 512, 256, 128, 64),
                             relu_leakiness: float = 0.2,
                             generate_intermediate_outputs: bool = False,
                             use_bilinear_upsampling: bool = True,
                             name: str = 'synthesis') -> tf.keras.Model:
  """Creates the synthesis network using the functional API.

  The function creates the synthesis network as defined in "A Style-Based
  Generator Architecture for Generative Adversarial Networks"
  https://arxiv.org/abs/1812.04948 using the Keras functional API.

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    upsampling_blocks_num_channels: The number of channels for each upsampling
      block. This argument also determines how many upsampling blocks are added.
    relu_leakiness: Slope of the negative part of the leaky relu.
    generate_intermediate_outputs: If true the model outputs a list of
      tf.Tensors with increasing resolution starting with the starting_size up
      to the final resolution output.
    use_bilinear_upsampling: If true bilinear upsampling is used.
    name: The name of the Keras model.

  Returns:
    The synthesis network.
  """

  kernel_initializer = tf.keras.initializers.TruncatedNormal(
      mean=0.0, stddev=1.0)

  mapped_latent_code_input = tf.keras.Input(shape=(latent_code_dimension,))

  tensor = keras_layers.LearnedConstant()(mapped_latent_code_input)
  tensor = keras_layers.Noise()(tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  tensor = apply_style_with_adain(
      mapped_latent_code=mapped_latent_code_input, input_tensor=tensor)
  tensor = keras_layers.FanInScaledConv2D(
      filters=upsampling_blocks_num_channels[0],
      kernel_size=3,
      strides=1,
      padding='same',
      kernel_initializer=kernel_initializer)(
          tensor)
  tensor = keras_layers.Noise()(tensor)
  tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
  tensor = apply_style_with_adain(
      mapped_latent_code=mapped_latent_code_input, input_tensor=tensor)

  outputs = []
  for index, channels in enumerate(upsampling_blocks_num_channels):
    if generate_intermediate_outputs:
      outputs.append(
          architectures_progressive_gan.to_rgb(
              input_tensor=tensor,
              kernel_initializer=kernel_initializer,
              name='side_output_%d_conv' % index))
    tensor = keras_layers.TwoByTwoNearestNeighborUpSampling()(tensor)
    if use_bilinear_upsampling:
      tensor = keras_layers.Blur2D()(tensor)
    for _ in range(2):
      tensor = keras_layers.FanInScaledConv2D(
          filters=channels,
          kernel_size=3,
          strides=1,
          padding='same',
          kernel_initializer=kernel_initializer)(
              tensor)
      tensor = keras_layers.Noise()(tensor)
      tensor = tf.keras.layers.LeakyReLU(alpha=relu_leakiness)(tensor)
      tensor = apply_style_with_adain(
          mapped_latent_code=mapped_latent_code_input, input_tensor=tensor)

  tensor = architectures_progressive_gan.to_rgb(
      input_tensor=tensor,
      kernel_initializer=kernel_initializer,
      name='final_output')
  if generate_intermediate_outputs:
    outputs.append(tensor)

    return tf.keras.Model(
        inputs=mapped_latent_code_input, outputs=outputs, name=name)
  else:
    return tf.keras.Model(
        inputs=mapped_latent_code_input, outputs=tensor, name=name)


def create_style_based_generator(
    latent_code_dimension: int = 128,
    upsampling_blocks_num_channels: Sequence[int] = (512, 256, 128, 64),
    relu_leakiness: float = 0.2,
    generate_intermediate_outputs: bool = False,
    use_bilinear_upsampling: bool = True,
    normalize_latent_code: bool = True,
    name: str = 'style_gan_generator'
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
  """Creates a Keras model for the style based generator with functional API.

  This architecture is implemented according to the paper "A Style-Based
  Generator Architecture for Generative Adversarial Networks"
  https://arxiv.org/abs/1812.04948
  The intermediate outputs are optionally provided for the architecture of
  "MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis"
  https://arxiv.org/abs/1903.06048

  Args:
    latent_code_dimension: The number of dimensions in the latent code.
    upsampling_blocks_num_channels: The number of channels for each upsampling
      block. This argument also determines how many upsampling blocks are added.
    relu_leakiness: Slope of the negative part of the leaky relu.
    generate_intermediate_outputs: If true the model outputs a list of
      tf.Tensors with increasing resolution starting with the starting_size up
      to the final resolution output.
    use_bilinear_upsampling: If true bilinear upsampling is used.
    normalize_latent_code: If true the latent code is normalized to unit length
      before feeding it to the network.
    name: The name of the Keras model.

  Returns:
     Three Keras models. The whole generator, only the mapping network and only
     the synthesis network.
  """
  mapping_network = create_mapping_network(
      latent_code_dimension=latent_code_dimension,
      normalize_latent_code=normalize_latent_code,
      relu_leakiness=relu_leakiness)
  synthesis_network = create_synthesis_network(
      latent_code_dimension=latent_code_dimension,
      upsampling_blocks_num_channels=upsampling_blocks_num_channels,
      relu_leakiness=relu_leakiness,
      generate_intermediate_outputs=generate_intermediate_outputs,
      use_bilinear_upsampling=use_bilinear_upsampling)

  input_tensor = tf.keras.Input(shape=(latent_code_dimension,))
  mapped_latent_code = mapping_network(input_tensor)
  generated_images = synthesis_network(mapped_latent_code)
  generator = tf.keras.Model(
      inputs=input_tensor, outputs=generated_images, name=name)

  return generator, mapping_network, synthesis_network
