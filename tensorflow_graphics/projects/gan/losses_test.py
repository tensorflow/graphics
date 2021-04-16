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
"""Tests for gan.losses."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.projects.gan import losses
from tensorflow_graphics.util import test_case


class LossesTest(test_case.TestCase):

  def test_gradient_penalty_shape_correct(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Reshape((25,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = tf.ones(shape=(3, 5, 5))
    generated_data = tf.ones(shape=(3, 5, 5))

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator)

    self.assertAllEqual(tf.shape(gradient_penalty), (3,))

  def test_gradient_penalty_shape_correct_sequence_input(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Concatenate())
    discriminator.add(tf.keras.layers.Reshape((50,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = (tf.ones(shape=(3, 5, 5)), tf.ones(shape=(3, 5, 5)))
    generated_data = (tf.ones(shape=(3, 5, 5)), tf.ones(shape=(3, 5, 5)))

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator)

    self.assertAllEqual(tf.shape(gradient_penalty), (3,))

  def test_gradient_penalty_loss_positive(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Reshape((25,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = tf.ones(shape=(1, 5, 5))
    generated_data = tf.ones(shape=(1, 5, 5))

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator)

    self.assertAllGreaterEqual(gradient_penalty, 0.0)

  def test_gradient_penalty_loss_positive_for_sequence_input(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Concatenate())
    discriminator.add(tf.keras.layers.Reshape((50,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = (tf.ones(shape=(1, 5, 5)), tf.ones(shape=(1, 5, 5)))
    generated_data = (tf.ones(shape=(1, 5, 5)), tf.ones(shape=(1, 5, 5)))

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator)

    self.assertAllGreaterEqual(gradient_penalty, 0.0)

  def test_gradient_penalty_loss_jacobian_preset(self):
    layer_weights = np.zeros(shape=(25, 1), dtype=np.float32)
    real_data = np.ones(shape=(1, 5, 5), dtype=np.float32)
    generated_data = np.ones(shape=(1, 5, 5), dtype=np.float32)

    def gradient_penalty_fn(weights):

      def multiply(input_tensor):
        return tf.linalg.matmul(input_tensor, weights)

      discriminator = tf.keras.Sequential()
      discriminator.add(tf.keras.layers.Reshape((25,)))
      # To simulate a dense layer a lambda layer is used, such that we are able
      # to feed the weights in as numpy array to the assert_jacobian_fn.
      discriminator.add(tf.keras.layers.Lambda(multiply))

      return losses.gradient_penalty_loss(
          real_data=tf.convert_to_tensor(real_data),
          generated_data=tf.convert_to_tensor(generated_data),
          discriminator=discriminator)

    with self.subTest(name='is_correct'):
      self.assert_jacobian_is_correct_fn(gradient_penalty_fn, (layer_weights,))
    with self.subTest(name='is_finite'):
      self.assert_jacobian_is_finite_fn(gradient_penalty_fn, (layer_weights,))

  def test_gradient_penalty_loss_sequence_input_jacobian_preset(self):
    layer_weights = np.zeros(shape=(50, 1), dtype=np.float32)
    real_data = (tf.ones(shape=(1, 5, 5), dtype=tf.float32),
                 tf.ones(shape=(1, 5, 5), dtype=tf.float32))
    generated_data = (tf.ones(shape=(1, 5, 5), dtype=tf.float32),
                      tf.ones(shape=(1, 5, 5), dtype=tf.float32))

    def gradient_penalty_fn(weights):

      def multiply(input_tensor):
        return tf.linalg.matmul(input_tensor, weights)

      discriminator = tf.keras.Sequential()
      discriminator.add(tf.keras.layers.Concatenate())
      discriminator.add(tf.keras.layers.Reshape((50,)))
      # To simulate a dense layer a lambda layer is used, such that we are able
      # to feed the weights in as numpy array to the assert_jacobian_fn.
      discriminator.add(tf.keras.layers.Lambda(multiply))

      return losses.gradient_penalty_loss(
          real_data=real_data,
          generated_data=generated_data,
          discriminator=discriminator)

    with self.subTest(name='is_correct'):
      self.assert_jacobian_is_correct_fn(gradient_penalty_fn, (layer_weights,))
    with self.subTest(name='is_finite'):
      self.assert_jacobian_is_finite_fn(gradient_penalty_fn, (layer_weights,))

  def test_gradient_penalty_loss_lambda_for_zero_gradient(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Reshape((4,)))
    # Generates a dense layer that is initialized with all zeros.
    # This leads to a network that has zero gradient for any input.
    discriminator.add(
        tf.keras.layers.Dense(
            units=1, kernel_initializer='zeros', bias_initializer='zeros'))
    real_data = tf.ones(shape=(1, 2, 2))
    generated_data = tf.ones(shape=(1, 2, 2))
    weight = 1.0

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator,
        weight=weight)

    # Tolerance is large due to eps that is added in the gradient pentaly loss
    # for numerical stability at 0.
    self.assertAllClose(gradient_penalty, (weight,), atol=0.001)

  def test_gradient_penalty_loss_lambda_for_zero_gradient_sequence_input(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Concatenate())
    discriminator.add(tf.keras.layers.Reshape((8,)))
    # Generates a dense layer that is initialized with all zeros.
    # This leads to a network that has zero gradient for any input.
    discriminator.add(
        tf.keras.layers.Dense(
            units=1, kernel_initializer='zeros', bias_initializer='zeros'))
    real_data = [tf.ones(shape=(1, 2, 2)), tf.ones(shape=(1, 2, 2))]
    generated_data = [tf.ones(shape=(1, 2, 2)), tf.ones(shape=(1, 2, 2))]
    weight = 1.0

    gradient_penalty = losses.gradient_penalty_loss(
        real_data=real_data,
        generated_data=generated_data,
        discriminator=discriminator,
        weight=weight)

    # Tolerance is large due to eps that is added in the gradient pentaly loss
    # for numerical stability at 0.
    self.assertAllClose(gradient_penalty, (weight,), atol=0.001)

  def test_gradient_penalty_loss_with_wrong_input_types_raises(self):
    discriminator = tf.keras.Sequential()

    with self.assertRaisesRegex(
        TypeError, 'should either both be a tf.Tensor '
        'or both a sequence of tf.Tensor'):
      losses.gradient_penalty_loss(
          real_data=(tf.ones((1,)),),
          generated_data=tf.ones((1,)),
          discriminator=discriminator)

  def test_gradient_penalty_loss_with_unequal_number_of_elements_raises(self):
    discriminator = tf.keras.Sequential()

    with self.assertRaisesRegex(
        ValueError, 'number of elements in real_data and generated_data are '
        'expected to be equal'):
      losses.gradient_penalty_loss(
          real_data=(tf.ones((1,)),),
          generated_data=(tf.ones((1,)), tf.ones((1,))),
          discriminator=discriminator)

  def test_r1_regularization_shape_correct(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Reshape((25,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = tf.ones(shape=(3, 5, 5))

    r1_regularization = losses.r1_regularization(
        real_data=real_data, discriminator=discriminator)
    r1_regularization_value = self.evaluate(r1_regularization)

    self.assertSequenceEqual(r1_regularization_value.shape, (3,))

  def test_r1_regularization_shape_correct_sequence_input(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Concatenate())
    discriminator.add(tf.keras.layers.Reshape((50,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = (tf.ones(shape=(3, 5, 5)), tf.ones(shape=(3, 5, 5)))

    r1_regulatiztion = losses.r1_regularization(
        real_data=real_data, discriminator=discriminator)
    r1_regulatiztion_value = self.evaluate(r1_regulatiztion)

    self.assertSequenceEqual(r1_regulatiztion_value.shape, (3,))

  def test_r1_regularization_positive(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Reshape((25,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = tf.ones(shape=(1, 5, 5))

    r1_regularization = losses.r1_regularization(
        real_data=real_data, discriminator=discriminator)

    self.assertAllGreaterEqual(r1_regularization, 0.0)

  def test_r1_regularization_positive_for_sequence_input(self):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Concatenate())
    discriminator.add(tf.keras.layers.Reshape((50,)))
    discriminator.add(tf.keras.layers.Dense(units=1))
    real_data = (tf.ones(shape=(1, 5, 5)), tf.ones(shape=(1, 5, 5)))

    r1_regularization = losses.r1_regularization(
        real_data=real_data, discriminator=discriminator)

    self.assertAllGreaterEqual(r1_regularization, 0.0)

  def test_r1_regularization_jacobian_random(self):
    layer_weights = np.random.uniform(-1, 1, size=(25, 1)).astype(np.float32)
    real_data = np.ones(shape=(1, 5, 5), dtype=np.float32)

    def r1_regularization_fn(weights):

      def multiply(input_tensor):
        return tf.linalg.matmul(input_tensor, weights)

      discriminator = tf.keras.Sequential()
      discriminator.add(tf.keras.layers.Reshape((25,)))
      # To simulate a dense layer a lambda layer is used, such that we are able
      # to feed the weights in as numpy array to the assert_jacobian_fn.
      discriminator.add(tf.keras.layers.Lambda(multiply))

      return losses.r1_regularization(
          real_data=tf.convert_to_tensor(real_data),
          discriminator=discriminator)

    with self.subTest(name='is_correct'):
      self.assert_jacobian_is_correct_fn(
          r1_regularization_fn, (layer_weights,), delta=0.001, atol=0.01)
    with self.subTest(name='is_finite'):
      self.assert_jacobian_is_finite_fn(r1_regularization_fn, (layer_weights,))

  def test_r1_regulatization_sequence_input_jacobian_random(self):
    layer_weights = np.random.uniform(-1, 1, size=(50, 1)).astype(np.float32)
    real_data = (tf.ones(shape=(1, 5, 5), dtype=tf.float32),
                 tf.ones(shape=(1, 5, 5), dtype=tf.float32))

    def r1_regulatization_fn(weights):

      def multiply(input_tensor):
        return tf.linalg.matmul(input_tensor, weights)

      discriminator = tf.keras.Sequential()
      discriminator.add(tf.keras.layers.Concatenate())
      discriminator.add(tf.keras.layers.Reshape((50,)))
      # To simulate a dense layer a lambda layer is used, such that we are able
      # to feed the weights in as numpy array to the assert_jacobian_fn.
      discriminator.add(tf.keras.layers.Lambda(multiply))

      return losses.r1_regularization(
          real_data=real_data, discriminator=discriminator)

    with self.subTest(name='is_correct'):
      self.assert_jacobian_is_correct_fn(
          r1_regulatization_fn, (layer_weights,), delta=0.001, atol=0.01)
    with self.subTest(name='is_finite'):
      self.assert_jacobian_is_finite_fn(r1_regulatization_fn, (layer_weights,))

  def test_wasserstein_generator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(losses.wasserstein_generator_loss(loss_input))

    self.assertAllEqual(loss.shape, (2, 1))

  @parameterized.parameters((losses.wasserstein_generator_loss, 0.0),
                            (losses.wasserstein_hinge_generator_loss, 0.0),
                            (losses.minimax_generator_loss, 0.0))
  def test_generator_loss_jacobian_preset(self, loss_function,
                                          loss_input_value):
    loss_input_init = np.full(
        shape=(2, 3), fill_value=loss_input_value, dtype=np.float32)
    loss_input = tf.convert_to_tensor(value=loss_input_init)

    loss = loss_function(loss_input)

    with self.subTest(name='is_finite'):
      self.assert_jacobian_is_finite(loss_input, loss_input_init, loss)
    with self.subTest(name='is_correct'):
      self.assert_jacobian_is_correct(
          loss_input, loss_input_init, loss, delta=1e-4, atol=1e-3)

  def test_wasserstein_discriminator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(
        losses.wasserstein_discriminator_loss(loss_input, loss_input))

    self.assertAllEqual(loss.shape, (2, 1))

  def test_wasserstein_discriminator_loss_zero_with_same_input(self):
    loss_input = tf.ones(shape=(5, 1))

    loss = self.evaluate(
        losses.wasserstein_discriminator_loss(loss_input, loss_input))

    self.assertAllClose(tf.reduce_sum(loss), 0.0)

  @parameterized.parameters(
      (losses.wasserstein_discriminator_loss, 0.0, 0.0),
      (losses.wasserstein_discriminator_loss, 0.5, 0.5),
      (losses.wasserstein_hinge_discriminator_loss, 1.0, -1.0),
      (losses.wasserstein_hinge_discriminator_loss, 0.0, 0.0),
      (losses.wasserstein_hinge_discriminator_loss, 2.0, -2.0),
      (losses.minimax_discriminator_loss, 0.0, 0.0))
  def test_discriminator_loss_jacobian_finite_preset(
      self, loss_function, discriminator_value_real,
      discriminator_value_generated):
    discriminator_value_real_init = np.full(
        shape=(2, 4), fill_value=discriminator_value_real, dtype=np.float32)
    discriminator_value_generated_init = np.full(
        shape=(2, 4),
        fill_value=discriminator_value_generated,
        dtype=np.float32)
    discriminator_value_real = tf.convert_to_tensor(
        value=discriminator_value_real_init)
    discriminator_value_generated = tf.convert_to_tensor(
        value=discriminator_value_generated_init)

    loss = loss_function(discriminator_value_real,
                         discriminator_value_generated)

    with self.subTest(name='with_respect_to_real'):
      self.assert_jacobian_is_finite(discriminator_value_real,
                                     discriminator_value_real_init, loss)
    with self.subTest(name='with_respcet_to_generated'):
      self.assert_jacobian_is_finite(discriminator_value_generated,
                                     discriminator_value_generated_init, loss)

  @parameterized.parameters(
      (losses.wasserstein_discriminator_loss, 0.0, 0.0),
      (losses.wasserstein_discriminator_loss, 0.5, 0.5),
      (losses.wasserstein_hinge_discriminator_loss, 0.0, 0.0),
      (losses.wasserstein_hinge_discriminator_loss, 2.0, -2.0),
      (losses.minimax_discriminator_loss, 0.0, 0.0))
  def test_discriminator_loss_jacobian_correct_preset(
      self, loss_function, discriminator_value_real,
      discriminator_value_generated):
    discriminator_value_real_init = np.full(
        shape=(2, 4), fill_value=discriminator_value_real, dtype=np.float32)
    discriminator_value_generated_init = np.full(
        shape=(2, 4),
        fill_value=discriminator_value_generated,
        dtype=np.float32)
    discriminator_value_real = tf.convert_to_tensor(
        value=discriminator_value_real_init)
    discriminator_value_generated = tf.convert_to_tensor(
        value=discriminator_value_generated_init)

    loss = loss_function(discriminator_value_real,
                         discriminator_value_generated)

    with self.subTest(name='with_respect_to_real'):
      self.assert_jacobian_is_correct(
          discriminator_value_real,
          discriminator_value_real_init,
          loss,
          delta=1e-4,
          atol=1e-3)
    with self.subTest(name='with_respcet_to_generated'):
      self.assert_jacobian_is_correct(
          discriminator_value_generated,
          discriminator_value_generated_init,
          loss,
          delta=1e-4,
          atol=1e-3)

  def test_wasserstein_hinge_generator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(losses.wasserstein_hinge_generator_loss(loss_input))

    self.assertAllEqual(loss.shape, (2, 1))

  def test_wasserstein_hinge_discriminator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(
        losses.wasserstein_hinge_discriminator_loss(loss_input, loss_input))

    self.assertAllEqual(loss.shape, (2, 1))

  @parameterized.parameters((1.0, 2.0, 3.0), (-1.0, 2.0, 5.0), (0.0, 2.0, 4.0),
                            (4.0, 3.0, 4.0), (-4.0, 3.0, 9.0), (4.0, -3.0, 0.0))
  def test_wasserstein_hinge_discriminator_loss_correct_value(
      self, real_data_input, generated_data_input, expected_loss_value):
    real_data_input = tf.fill(dims=(), value=real_data_input)
    generated_data_input = tf.fill(dims=(), value=generated_data_input)

    loss = self.evaluate(
        losses.wasserstein_hinge_discriminator_loss(real_data_input,
                                                    generated_data_input))

    self.assertAlmostEqual(loss, expected_loss_value)

  def test_minimax_generator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(losses.minimax_generator_loss(loss_input))

    self.assertAllEqual(loss.shape, (2, 1))

  def test_minimax_discriminator_loss_shape_correct(self):
    loss_input = tf.ones(shape=(2, 1))

    loss = self.evaluate(
        losses.minimax_discriminator_loss(loss_input, loss_input))

    self.assertAllEqual(loss.shape, (2, 1))


if __name__ == '__main__':
  tf.test.main()
