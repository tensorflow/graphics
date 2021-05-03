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
"""Module with loss functions."""

import collections
from typing import Sequence, Union

import tensorflow as tf


def gradient_penalty_loss(real_data: Union[tf.Tensor, Sequence[tf.Tensor]],
                          generated_data: Union[tf.Tensor, Sequence[tf.Tensor]],
                          discriminator: tf.keras.Model,
                          weight: float = 10.0,
                          eps: float = 1e-8,
                          name_scope: str = 'gradient_penalty') -> tf.Tensor:
  """Gradient penalty loss.

  This function implements the gradient penalty loss from "Improved Training of
  Wasserstein GANs"
  https://arxiv.org/abs/1704.00028
  This version also implements the multi-scale extension proposed in "MSG-GAN:
  Multi-Scale Gradient GAN for Stable Image Synthesis"
  https://arxiv.org/abs/1903.06048

  Args:
    real_data: Samples from the real data.
    generated_data: Samples from the generated data.
    discriminator: The Keras model of the discriminator. This model is expected
      to take as input a tf.Tensor or a sequence of tf.Tensor depending on what
      is provided as generated_data and real_data.
    weight: The weight of the loss.
    eps: A small positive value that is added to the argument of the square root
      to avoid undefined gradients.
    name_scope: The name scope of the loss.

  Returns:
    The gradient penalty loss.

  Raises:
    TypeError if real_data and generated_data are not both either tf.Tensor or
    both a sequence of tf.Tensor.
    ValueError if the numnber of elements in real_data and generated_data are
    not equal.
  """
  with tf.name_scope(name=name_scope):
    with tf.GradientTape() as tape:
      if (isinstance(real_data, tf.Tensor) and
          isinstance(generated_data, tf.Tensor)):
        epsilon = tf.random.uniform(
            [tf.shape(real_data)[0]] + [1] * (real_data.shape.ndims - 1),
            minval=0.0,
            maxval=1.0,
            dtype=real_data.dtype)
        interpolated_data = epsilon * real_data + (1.0 -
                                                   epsilon) * generated_data
      elif (isinstance(real_data, collections.Sequence) and
            isinstance(generated_data, collections.Sequence)):
        if len(real_data) != len(generated_data):
          raise ValueError(
              'The number of elements in real_data and generated_data are '
              'expected to be equal but got: %d and %d' %
              (len(real_data), len(generated_data)))
        epsilon = tf.random.uniform(
            [tf.shape(real_data[0])[0]] + [1] * (real_data[0].shape.ndims - 1),
            minval=0.0,
            maxval=1.0,
            dtype=real_data[0].dtype)
        interpolated_data = [
            epsilon * real_level + (1.0 - epsilon) * generated_level
            for real_level, generated_level in zip(real_data, generated_data)
        ]
      else:
        raise TypeError(
            'real_data and generated data should either both be a tf.Tensor '
            'or both a sequence of tf.Tensor but got: %s and %s' %
            (type(real_data), type(generated_data)))
      # By default the gradient tape only watches trainable variables.
      tape.watch(interpolated_data)
      interpolated_labels = discriminator(interpolated_data)

      with tf.name_scope(name='gradients'):
        gradients = tape.gradient(
            target=interpolated_labels, sources=interpolated_data)

    if isinstance(real_data, tf.Tensor):
      gradient_squares = tf.reduce_sum(
          input_tensor=tf.square(gradients),
          axis=tuple(range(1, gradients.shape.ndims)))

      gradient_norms = tf.sqrt(gradient_squares + eps)
      penalties_squared = tf.square(gradient_norms - 1.0)
      return weight * penalties_squared
    else:
      all_penalties_squared = []
      for gradients_level in gradients:
        gradient_squares_level = tf.reduce_sum(
            input_tensor=tf.square(gradients_level),
            axis=tuple(range(1, gradients_level.shape.ndims)))

        gradient_norms_level = tf.sqrt(gradient_squares_level + eps)
        all_penalties_squared.append(tf.square(gradient_norms_level - 1.0))

      return weight * tf.add_n(all_penalties_squared) * 1.0 / len(real_data)


def _sum_of_squares(input_tensor):
  """Computes the sum of squares of the tensor for each element in the batch."""
  return tf.reduce_sum(
      input_tensor=tf.square(input_tensor),
      axis=tuple(range(1, input_tensor.shape.ndims)))


def r1_regularization(real_data: Union[tf.Tensor, Sequence[tf.Tensor]],
                      discriminator: tf.keras.Model,
                      weight: float = 10.0,
                      name='r1_regularization') -> tf.Tensor:
  """Implements the r1 regulariztion loss.

  This regularization loss for discriminators is proposed in "Which Training
  Methods for GANs do actually Converge?"
  https://arxiv.org/abs/1801.04406
  This version also implements the multi-scale extension proposed in "MSG-GAN:
  Multi-Scale Gradient GAN for Stable Image Synthesis"
  https://arxiv.org/abs/1903.06048

  Args:
    real_data: Samples from the real data.
    discriminator: The Keras model of the discriminator. This model is expected
      to take as input a tf.Tensor or a sequence of tf.Tensor depending on what
      is provided as real_data.
    weight: The weight of the loss.
    name: The name scope of the loss.

  Returns:
    The r1 regulatization loss per example as tensor of shape [batch_size].
  """
  with tf.name_scope(name):
    with tf.GradientTape() as tape:
      tape.watch(real_data)
      discriminator_output = discriminator(real_data)

      with tf.name_scope(name='gradients'):
        gradients = tape.gradient(
            target=discriminator_output, sources=real_data)

    if isinstance(real_data, tf.Tensor):
      gradient_squares = _sum_of_squares(gradients)

      return weight * 0.5 * gradient_squares
    else:
      gradient_squares_level = [
          _sum_of_squares(gradients_level) for gradients_level in gradients
      ]

      return weight * 0.5 * tf.add_n(gradient_squares_level) * 1.0 / len(
          real_data)


def wasserstein_generator_loss(
    discriminator_output_generated_data: tf.Tensor,
    name: str = 'wasserstein_generator_loss') -> tf.Tensor:
  """Generator loss for Wasserstein GAN.

  This loss function is generally used together with a regularization of the
  discriminator such as weight clipping (https://arxiv.org/abs/1701.07875),
  gradient penalty (https://arxiv.org/abs/1704.00028) or spectral
  normalization (https://arxiv.org/abs/1802.05957).

  Args:
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the generator.
  """
  with tf.name_scope(name=name):
    return -discriminator_output_generated_data


def wasserstein_discriminator_loss(
    discriminator_output_real_data: tf.Tensor,
    discriminator_output_generated_data: tf.Tensor,
    name: str = 'wasserstein_discriminator_loss') -> tf.Tensor:
  """Discriminator loss for Wasserstein GAN.

  This loss function is generally used together with a regularization of the
  discriminator such as weight clipping (https://arxiv.org/abs/1701.07875),
  gradient penalty (https://arxiv.org/abs/1704.00028) or spectral
  normalization (https://arxiv.org/abs/1802.05957).

  Args:
    discriminator_output_real_data: Output of the discriminator for the real
      data.
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the discriminator.
  """
  with tf.name_scope(name=name):
    return discriminator_output_generated_data - discriminator_output_real_data


def wasserstein_hinge_generator_loss(
    discriminator_output_generated_data: tf.Tensor,
    name: str = 'wasserstein_hinge_generator_loss') -> tf.Tensor:
  """Generator loss for the hinge Wasserstein GAN.

  This loss function is generally used together with a regularization of the
  discriminator such as weight clipping (https://arxiv.org/abs/1701.07875),
  gradient penalty (https://arxiv.org/abs/1704.00028) or spectral
  normalization (https://arxiv.org/abs/1802.05957).
  Note that the generator loss does not have a hinge
  (https://arxiv.org/pdf/1805.08318.pdf).

  Args:
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the generator.
  """
  with tf.name_scope(name=name):
    return -discriminator_output_generated_data


def wasserstein_hinge_discriminator_loss(
    discriminator_output_real_data: tf.Tensor,
    discriminator_output_generated_data: tf.Tensor,
    name: str = 'wasserstein_hinge_discriminator_loss') -> tf.Tensor:
  """Discriminator loss for the hinge Wasserstein GAN.

  This loss function is generally used together with a regularization of the
  discriminator such as weight clipping (https://arxiv.org/abs/1701.07875),
  gradient penalty (https://arxiv.org/abs/1704.00028) or spectral
  normalization (https://arxiv.org/abs/1802.05957).

  Args:
    discriminator_output_real_data: Output of the discriminator for the real
      data.
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the discriminator.
  """
  with tf.name_scope(name=name):
    return tf.nn.relu(1.0 - discriminator_output_real_data) + tf.nn.relu(
        discriminator_output_generated_data + 1.0)


def minimax_generator_loss(discriminator_output_generated_data: tf.Tensor,
                           name: str = 'minimax_generator_loss') -> tf.Tensor:
  """Generator loss from the original minimax GAN.

  This loss function implements the non saturating version of the minimax loss
  for the generator as proposed in https://arxiv.org/pdf/1406.2661.pdf.

  Args:
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the generator.
  """
  with tf.name_scope(name=name):
    # -log(sigmoid(discriminator_output_generated_data))
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(discriminator_output_generated_data),
        logits=discriminator_output_generated_data)


def minimax_discriminator_loss(
    discriminator_output_real_data: tf.Tensor,
    discriminator_output_generated_data: tf.Tensor,
    name: str = 'minimax_discriminator_loss') -> tf.Tensor:
  """Discriminator loss from the original minimax GAN.

  This loss function implements the minimax loss
  for the discriminator as proposed in https://arxiv.org/pdf/1406.2661.pdf.

  Args:
    discriminator_output_real_data: Output of the discriminator for the real
      data.
    discriminator_output_generated_data: Output of the discriminator for
      generated data.
    name: The name of the name_scope that is placed around the loss.

  Returns:
    The loss for the discriminator.
  """
  with tf.name_scope(name=name):
    # -log(sigmoid(discriminator_output_real_data))
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(discriminator_output_real_data),
        logits=discriminator_output_real_data)
    # -log(1 - sigmoid(discriminator_output_generated_data))
    loss_generated = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(discriminator_output_generated_data),
        logits=discriminator_output_generated_data)
    return loss_real + loss_generated
