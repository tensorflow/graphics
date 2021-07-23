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
"""Keras layers for gan architectures."""

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import tensorflow as tf

_InitializerCallable = Callable[[tf.Tensor, tf.dtypes.DType], tf.Tensor]
_KerasInitializer = Union[_InitializerCallable, str]


class _KernelFanInScaler(tf.keras.layers.Layer):
  """Scales the kernel weights with sqrt(multiplier/fan_in)*kernel_multiplier.

  This class is meant to be used as a base class to inherit from in conjunction
  with also inheriting from an existing keras layer. It assumes that
  the tf.Variable containing the kernel weights is assigned to self.kernel and
  that the fan_in is the product of the kernel shape excluding the last
  dimension. It also allows for optionally scaling the bias for the equalized
  learning rate and hence expects that the existing keras layer has a self.bias
  variable in this case.

  Layers built with this class scale the kernel with the same scaling factor
  that is proposed in "Delving Deep into Rectifiers: Surpassing Human-Level
  Performance on ImageNet Classification" for initialization
  https://arxiv.org/abs/1502.01852.
  The kernel gets scaled by sqrt(2/fan_in) before it is being used.

  These layers can be used for implementing the equalized learning rate which is
  used in "Progressive Growing of GANs for Improved Quality, Stability, and
  Variation" https://arxiv.org/abs/1710.10196.

  The layers also support specifying additional multipliers which can be used to
  implement a learning rate multiplier if ADAM or other adaptive optimizers are
  used. This mechanism can be used to implement the learning rate multiplier of
  the mapping network in stylegan.
  https://arxiv.org/abs/1812.04948
  """

  def __init__(self,
               *args,
               kernel_multiplier: float = 1.0,
               bias_multiplier: Optional[float] = None,
               multiplier: float = math.sqrt(2.0),
               **kwargs):
    """Initializes the layer.

    Args:
      *args: Arguments that get forwarded to the super initializer.
      kernel_multiplier: A multiplier of the kernel. This can be used as a per
        layer learning rate multiplier.
      bias_multiplier: A multiplier of the bias. This can be used as per layer
        learning rate multiplier.
      multiplier: The kernel is multiplied by multiplier/sqrt(fan_in). The
        default value is sqrt(2.0) and is used to compensate for the effects of
        the ReLU. See https://arxiv.org/abs/1502.01852 for details.
      **kwargs: Keyword argument dictionary that gets forwarded to the super
        initializer.
    """
    self._kernel_multiplier = kernel_multiplier
    self._bias_multiplier = bias_multiplier
    self._multiplier = multiplier
    super().__init__(*args, **kwargs)

  @property
  def kernel(self) -> tf.Tensor:
    # Fan in computation assumes that the last dimension of the kernel is the
    # output dimension.
    fan_in = np.prod(self._kernel.shape.as_list()[:-1])
    return (self._kernel * self._multiplier / math.sqrt(fan_in) *
            self._kernel_multiplier)

  @kernel.setter
  def kernel(self, kernel: tf.Variable) -> None:
    self._kernel = kernel

  @property
  def bias(self) -> tf.Tensor:
    if self._bias_multiplier is not None:
      return self._bias * self._bias_multiplier
    else:
      return self._bias.read_value()

  @bias.setter
  def bias(self, bias: tf.Variable) -> None:
    self._bias = bias

  def get_config(self):
    """Gets the configuration dictionary of the layer."""
    config = {
        'kernel_multiplier': self._kernel_multiplier,
        'bias_multiplier': self._bias_multiplier,
        'multiplier': self._multiplier
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FanInScaledDense(_KernelFanInScaler, tf.keras.layers.Dense):
  """Dense layer with the kernel scaled with sqrt(2/fan_in)*kernel_multiplier.

  This layer can be used to implement the equalized learning rate proposed in
  "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
  https://arxiv.org/abs/1710.10196
  """
  pass


class FanInScaledConv2D(_KernelFanInScaler, tf.keras.layers.Conv2D):
  """Conv2D layer with the kernel scaled with sqrt(2/fan_in)*kernel_multiplier.

  This layer can be used to implement the equalized learning rate proposed in
  "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
  https://arxiv.org/abs/1710.10196
  """
  pass


class PixelNormalization(tf.keras.layers.Layer):
  """The pixel normalization layer.

  This layer normalizes each feature vector of the feature map to 'unit' length
  as described in the paper:
  "Progressive growing of GANs for Improved Quality, Stability, and Variation"
  https://arxiv.org/pdf/1710.10196.pdf
  The per pixel length of the features is sqrt(C) with C the number of channels.
  """

  def __init__(self, axis: int, epsilon: float = 1e-8, **kwargs):
    """Initializes the pixel normalization layer.

    Args:
      axis: The axis that corresponds to the feature channels over which the
        normalization should take place.
      epsilon: The epslion that is added to aviod numerical stability issues
        with the divion.
      **kwargs: Dictionary with additional keyword arguments that get forwarded
        to the super class initializer.
    """
    super().__init__(**kwargs)
    self._axis = axis
    self._epsilon = epsilon

  def call(self, inputs):
    """The call function that does the actual normalization."""
    return inputs * tf.math.rsqrt(
        tf.reduce_mean(tf.square(inputs), axis=self._axis, keepdims=True) +
        self._epsilon)

  def get_config(self):
    """Returns the config of the layer."""
    config = {'axis': self._axis, 'epsilon': self._epsilon}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TwoByTwoNearestNeighborUpSampling(tf.keras.layers.Layer):
  """A two by two nearest neighbor Keras upsampling layer.

  Each spatial location is expanded to at 2x2 region by replicating the same
  feature 4 times.
  """

  def call(self, inputs):
    """The call function that does the actual upsampling."""
    shape = tf.shape(inputs)
    # Number of channels needs to be used from the static shape because
    # following convolutional layers need to know the numer of channels
    # statically. The rest can be dynamic.
    number_of_channels = inputs.shape[3]
    reshaped = inputs[:, :, tf.newaxis, :, tf.newaxis, :]
    upsampled = tf.tile(reshaped, (1, 1, 1, 2, 1, 2))
    return tf.reshape(
        upsampled, (shape[0], shape[1] * 2, shape[2] * 2, number_of_channels))


class Blur2D(tf.keras.layers.Layer):
  """The blur layer applies a [1, 2, 1] blur to the input channels.

  If such a filter is applied after nearest neighbor upsampling it results in
  implementing bilinear upsamplig ([1, 3, 3, 1] kernel on bed of nails
  upsampling). Similarly this can be used to implement the analoguos
  anti-aliased bilinear downsampling ([1, 3, 3, 1] kernel) if applied to the
  input feature map before average pooling.
  """

  def __init__(self, **kwargs):
    """Initializes the blur layer."""
    super().__init__(**kwargs)
    self._filter_sequence = np.array((1, 2, 1), dtype=np.float32)

  def build(self, input_shape: tf.TensorShape):
    """Builds the blur layer."""
    single_kernel = np.outer(self._filter_sequence, self._filter_sequence)
    single_kernel /= np.sum(single_kernel)
    single_kernel = tf.convert_to_tensor(single_kernel)[..., tf.newaxis,
                                                        tf.newaxis]
    self._kernel = tf.tile(single_kernel, (1, 1, input_shape[-1], 1))
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the blur layer."""
    return tf.nn.depthwise_conv2d(inputs, self._kernel, (1, 1, 1, 1), 'SAME')


class LearnedConstant(tf.keras.layers.Layer):
  """Learned constant layer for StyleGAN.

  This layer implements the learned constant that is used in "A Style-Based
  Generator Architecture for Generative Adversarial Networks".
  https://arxiv.org/pdf/1812.04948.pdf
  """

  def __init__(self, shape: Sequence[int] = (4, 4, 512), **kwargs):
    """Initialized the learned constant layer."""
    super().__init__(**kwargs)
    self._shape = shape

  def build(self, input_shape: tf.TensorShape):
    """Adds the variable for the learned constant to the layer."""
    self._learned_constant = self.add_weight(
        name='learned_constant',
        shape=self._shape,
        initializer=tf.keras.initializers.get('ones'))
    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Returns the constant tiled along the batch axis to match inputs."""
    batch_size = tf.shape(inputs)[0]
    return tf.tile(
        self._learned_constant[tf.newaxis, ...],
        multiples=(batch_size, 1, 1, 1))

  def get_config(self):
    """Returns the config of the layer."""
    config = {'shape': self._shape}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Noise(tf.keras.layers.Layer):
  """Noise layer for StyleGAN.

  This layer can be used to inject the stochastic noise that is used in "A
  Style-Based Generator Architecture for Generative Adversarial Networks".
  https://arxiv.org/pdf/1812.04948.pdf

  It samples a single channel Gaussian noise image and broadcasts and adds it to
  the feature channels of the input using a learned per-feature channel scaling
  factor.
  """

  def build(
      self, input_shape: Union[tf.TensorShape,
                               Sequence[tf.TensorShape]]) -> None:
    if not isinstance(input_shape, tf.TensorShape):
      if len(input_shape) != 2:
        raise ValueError('Either a single input feature map or 2 inputs '
                         '(feature map and noise) are expected.')
      feature_map_shape, _ = input_shape
    else:
      feature_map_shape = input_shape

    self._per_channel_scales = self.add_weight(
        name='per_channel_scales',
        shape=(feature_map_shape[-1],),
        initializer=tf.keras.initializers.get('zeros'),
        trainable=True)
    super().build(input_shape)

  def call(self, inputs: Union[Sequence[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    """Returns noise with dimensions corresponding to the input.

    Args:
      inputs: A single tensor containing the feature_map to which random noise
        is added or two tensors containing the feature_map and the single
        channel noise.

    Returns:
      The feature map with the per channel scaled noise added.
    """
    if isinstance(inputs, tf.Tensor):
      batch_size, height, width = tf.unstack(tf.shape(inputs)[:3])
      noise = tf.random.truncated_normal(shape=(batch_size, height, width, 1))
      feature_map = inputs
    else:
      feature_map, noise = inputs
    return noise * self._per_channel_scales + feature_map


class DemodulatedConvolution(tf.keras.layers.Layer):
  """Demodulation layer for StyleGAN.

  This layer applies the Demodulated Convolution layer from StyleGANv2
  that is used in the paper "
  Analyzing and Improving the Image Quality of StyleGAN".
  https://arxiv.org/pdf/1912.04958.pdf

  It takes in the style input and applies demodulation on the features.
  """

  def __init__(self,
               filters: int,
               kernel_size: int,
               kernel_initializer: Optional[_KerasInitializer] = None,
               bias_initializer: _KerasInitializer = 'zeros',
               **kwargs: Any) -> None:
    """Initizlies the demodulated convolution layer.

    Args:
      filters: The number of filters the convolution will use.
      kernel_size: The kernel size of the convolution.
      kernel_initializer: The initializer of the kernels of the convolution and
        dense layer.
      bias_initializer: The initializer of the bias of the convolution and dense
        layer.
      **kwargs: The keyword arguments that are forwarded to the super class.
    """
    super().__init__(**kwargs)
    if kernel_initializer is None:
      self._kernel_initializer = tf.keras.initializers.TruncatedNormal(
          mean=0.0, stddev=1.0)
    else:
      self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    self._bias_initializer = tf.keras.initializers.get(bias_initializer)

    self._conv_layer = FanInScaledConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        use_bias=False)
    self._kernel_size = kernel_size
    self._filters = filters

  def build(self, input_shape: Sequence[tf.TensorShape]) -> None:
    feature_map_shape, _ = input_shape
    self._dense_layer = FanInScaledDense(
        feature_map_shape[-1],
        multiplier=1.0,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer)
    self._bias = self.add_weight(
        shape=(self._filters,),
        name='bias',
        initializer=self._bias_initializer,
        trainable=True)

  def call(
      self, inputs: Union[List[tf.Tensor], Tuple[tf.Tensor,
                                                 tf.Tensor]]) -> tf.Tensor:
    """Returns the demodaulted features.

    Args:
      inputs: Two elements, the input feature map and the mapped latent code.

    Returns:
      The feature map after applying the demodulated convolution.
    """
    if len(inputs) != 2:
      raise ValueError('inputs needs to have two elements.')

    feature_map, mapped_latent_code = inputs
    style = self._dense_layer(mapped_latent_code)[:, tf.newaxis, tf.newaxis, :]

    modulated_features = self._conv_layer(feature_map * style)
    weight = self._conv_layer.kernel[tf.newaxis, :, :, :, :]

    demodulation = tf.math.rsqrt(
        tf.math.reduce_sum((tf.math.square(
            weight * style[:, :, :, :, tf.newaxis])), [1, 2, 3]) + 1e-8)
    demodulated_features = modulated_features * demodulation[:, tf.newaxis,
                                                             tf.newaxis, :]

    return tf.nn.bias_add(demodulated_features, self._bias)

  def get_config(self) -> Dict[str, Any]:
    """Returns the config of the layer."""
    config = {
        'kernel_size':
            self._kernel_size,
        'filters':
            self._filters,
        'kernel_initializer':
            tf.keras.utils.serialize_keras_object(self._kernel_initializer),
        'bias_initializer':
            tf.keras.utils.serialize_keras_object(self._bias_initializer),
    }
    base_config = super().get_config()
    base_config.update(config)
    return base_config


# Dictionary that contains all the custom layers defined in this file.
# This can be used to register the custom layers with Keras, e.g.
# tf.keras.utils.custom_object_scope
CUSTOM_LAYERS = {
    'FanInScaledDense': FanInScaledDense,
    'FanInScaledConv2D': FanInScaledConv2D,
    'DemodulatedConvolution': DemodulatedConvolution,
    'PixelNormalization': PixelNormalization,
    'LearnedConstant': LearnedConstant,
    'Noise': Noise,
    'TwoByTwoNearestNeighborUpSampling': TwoByTwoNearestNeighborUpSampling,
    'Blur2D': Blur2D,
}
