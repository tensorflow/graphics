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
"""Tests for gan.keras_layers."""

import math

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.gan import keras_layers
from tensorflow_graphics.util import test_case


class FanInScaledDenseTest(test_case.TestCase):

  def test_kernel_shape_correct(self):
    dense_layer = keras_layers.FanInScaledDense(units=4)

    dense_layer(tf.ones((1, 2)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertSequenceEqual(dense_layer.kernel.shape, (2, 4))

  @parameterized.parameters(1.0, 0.01, 2.0)
  def test_kernel_has_correct_value(self, kernel_multiplier):
    dense_layer = keras_layers.FanInScaledDense(
        units=1, kernel_initializer='ones', kernel_multiplier=kernel_multiplier)

    dense_layer(tf.ones((1, 4)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Checking if the 1 is correctly multiplied with sqrt(2/fan_in).
    self.assertAllClose(dense_layer.kernel,
                        ((math.sqrt(2.0 / 4.0) * kernel_multiplier,),) * 4)

  def test_bias_has_correct_value(self, bias_multiplier=2.0):
    dense_layer = keras_layers.FanInScaledDense(
        units=1, bias_initializer='ones', bias_multiplier=bias_multiplier)

    dense_layer(tf.ones((1, 4)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose(dense_layer.bias, (bias_multiplier,))

  def test_config_contains_both_base_classes(self):
    conv2d_layer = keras_layers.FanInScaledDense(
        units=3, kernel_multiplier=2.0)

    config_dict = conv2d_layer.get_config()

    with self.subTest(name='_FanInScaler'):
      self.assertIn('kernel_multiplier', config_dict)
      self.assertIn('bias_multiplier', config_dict)
      self.assertEqual(config_dict['kernel_multiplier'], 2.0)
      self.assertIsNone(config_dict['bias_multiplier'])

    with self.subTest(name='Dense'):
      self.assertIn('units', config_dict)
      self.assertEqual(config_dict['units'], 3)


class FanInScaledConv2DTest(test_case.TestCase):

  def test_kernel_shape_correct(self):
    conv2d_layer = keras_layers.FanInScaledConv2D(
        filters=5, kernel_size=(2, 3))

    conv2d_layer(tf.ones((1, 8, 6, 4)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertSequenceEqual(conv2d_layer.kernel.shape, (2, 3, 4, 5))

  @parameterized.parameters(1.0, 0.01, 2.0)
  def test_kernel_has_correct_value(self, kernel_multiplier):
    conv2d_layer = keras_layers.FanInScaledConv2D(
        filters=5,
        kernel_size=(2, 3),
        kernel_initializer='ones',
        kernel_multiplier=kernel_multiplier)

    conv2d_layer(tf.ones((1, 8, 6, 4)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Checking if the 1 is correctly multiplied with sqrt(2/fan_in).
    self.assertAllClose(
        conv2d_layer.kernel,
        tf.ones(shape=(2, 3, 4, 5)) * math.sqrt(2.0 / (2.0 * 3.0 * 4.0)) *
        kernel_multiplier)

  def test_bias_has_correct_value(self, bias_multiplier=2.0):
    conv2d_layer = keras_layers.FanInScaledConv2D(
        filters=1,
        kernel_size=3,
        bias_initializer='ones',
        bias_multiplier=bias_multiplier)

    conv2d_layer(tf.ones((1, 3, 3, 4)))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose(tf.squeeze(conv2d_layer.bias), bias_multiplier)

  def test_config_contains_both_base_classes(self):
    conv2d_layer = keras_layers.FanInScaledConv2D(
        filters=3, kernel_size=1, kernel_multiplier=2.0, multiplier=1.0)

    config_dict = conv2d_layer.get_config()

    with self.subTest(name='_FanInScaler'):
      self.assertIn('kernel_multiplier', config_dict)
      self.assertIn('multiplier', config_dict)
      self.assertIn('bias_multiplier', config_dict)
      self.assertEqual(config_dict['kernel_multiplier'], 2.0)
      self.assertEqual(config_dict['multiplier'], 1.0)
      self.assertIsNone(config_dict['bias_multiplier'])

    with self.subTest(name='Conv2D'):
      self.assertIn('filters', config_dict)
      self.assertEqual(config_dict['filters'], 3)


class PixelNormalizationTest(test_case.TestCase):

  @parameterized.parameters(((2,), 0), ((2, 3, 4), 1), ((6, 3, 4), 2))
  def test_output_is_normalized(self, shape, axis):
    input_tensor = tf.random.normal(shape=shape)

    normalized_tensor = keras_layers.PixelNormalization(axis=axis)(input_tensor)
    normalized_lengths = tf.norm(normalized_tensor, axis=axis)

    self.assertAllClose(
        normalized_lengths,
        tf.fill(dims=normalized_lengths.shape, value=math.sqrt(shape[axis])))

  @parameterized.parameters(0, 3)
  def config_contains_correct_axis(self, axis):
    layer = keras_layers.PixelNormalization(axis=axis)

    config = layer.get_config()

    self.assertIn(config, 'axis')
    self.assertEqual(config['axis'], axis)


class TwoByTwoNearestNeighborUpSampling(tf.test.TestCase):

  def test_upsampling_is_correct(self, shape=(1, 2, 2, 3)):
    input_tensor = tf.random.normal(shape=shape)

    upsampled_tensor = keras_layers.TwoByTwoNearestNeighborUpSampling()(
        input_tensor)

    self.assertAllEqual(upsampled_tensor[:, ::2, ::2, :], input_tensor)
    self.assertAllEqual(upsampled_tensor[:, 1::2, ::2, :], input_tensor)
    self.assertAllEqual(upsampled_tensor[:, ::2, 1::2, :], input_tensor)
    self.assertAllEqual(upsampled_tensor[:, 1::2, 1::2, :], input_tensor)


class Blur2DTest(tf.test.TestCase):

  def test_upsampling_blur_is_bilinear_upsampling(self, shape=(1, 4, 4, 3)):
    input_tensor = tf.random.normal(shape=shape)

    upsample_tensor = keras_layers.TwoByTwoNearestNeighborUpSampling()(
        input_tensor)
    blurred_tensor = keras_layers.Blur2D()(upsample_tensor)

    height, width = shape[1:3]
    bilinear_interpolated_tensor = tf.image.resize(
        input_tensor, (height * 2, width * 2),
        method=tf.image.ResizeMethod.BILINEAR)
    self.assertAllClose(blurred_tensor[:, 1:-1, 1:-1, :],
                        bilinear_interpolated_tensor[:, 1:-1, 1:-1, :])

  def test_blur_average_pooling_is_anti_aliased_bilinear_downsampling(
      self, shape=(1, 8, 8, 3)):
    input_tensor = tf.random.normal(shape=shape)

    blurred_tensor = keras_layers.Blur2D()(input_tensor)
    average_pooled_tensor = tf.keras.layers.AveragePooling2D()(blurred_tensor)

    height, width = shape[1:3]
    bilinear_interpolated_tensor = tf.image.resize(
        input_tensor, (height // 2, width // 2),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True)
    self.assertAllClose(average_pooled_tensor[:, 1:-1, 1:-1, :],
                        bilinear_interpolated_tensor[:, 1:-1, 1:-1, :])


class LearnedConstantTest(tf.test.TestCase):

  def test_output_shape_correct(self, constant_shape=(4, 4, 6), batch_size=3):
    input_tensor = tf.zeros((batch_size,))

    output_tensor = keras_layers.LearnedConstant(shape=constant_shape)(
        input_tensor)

    self.assertSequenceEqual(output_tensor.shape,
                             (batch_size,) + constant_shape)

  def test_config_contains_correct_shape(self, constant_shape=(1, 5, 2)):
    layer = keras_layers.LearnedConstant(shape=constant_shape)

    config = layer.get_config()

    self.assertIn('shape', config)
    self.assertSequenceEqual(config['shape'], constant_shape)


class NoiseTest(tf.test.TestCase):

  def test_output_shape_correct(self, shape=(2, 5, 5, 3)):
    input_tensor = tf.ones(shape)

    output_tensor = keras_layers.Noise()(input_tensor)

    self.assertSequenceEqual(output_tensor.shape, shape)

  def test_output_shape_correct_with_input_noise(self, shape=(2, 5, 7, 3)):
    input_feature_map = tf.ones(shape)
    input_noise = tf.ones(shape[:-1] + (1,))

    output_tensor = keras_layers.Noise()((input_feature_map, input_noise))

    self.assertSequenceEqual(output_tensor.shape, shape)

  def test_raises_with_wrong_number_of_inputs(self):
    with self.assertRaisesRegex(ValueError,
                                'single input feature map or 2 inputs'):
      _ = keras_layers.Noise()(
          inputs=(tf.ones((1, 1)), tf.ones((1, 1)), tf.ones((1, 1))))


class DemodulatedConvolutionTest(tf.test.TestCase):

  def test_output_shape_correct(self):
    input_tensor = tf.ones(shape=(2, 5, 5, 16))
    style = tf.ones(shape=(2, 48))
    demod_layer = keras_layers.DemodulatedConvolution(filters=32, kernel_size=3)

    output_tensor = demod_layer([input_tensor, style])

    self.assertSequenceEqual(output_tensor.shape, (2, 5, 5, 32))

  def test_output_feature_map_has_correct_mean_and_stddev(self):
    input_tensor = tf.random.normal(shape=(2, 256, 256, 16))
    style = tf.ones(shape=(2, 48))

    demod_layer = keras_layers.DemodulatedConvolution(3, 32)

    output_tensor = demod_layer([input_tensor, style])
    mean, variance = tf.nn.moments(output_tensor, axes=[0, 1, 2, 3])
    self.assertAllClose(mean, 0, atol=1e-1)
    self.assertAllClose(variance, 1, atol=1e-1)

  def test_config_contains_correct_entries(self):
    layer = keras_layers.DemodulatedConvolution(filters=8, kernel_size=3)

    config = layer.get_config()

    self.assertContainsSubset(
        ('kernel_size', 'filters', 'kernel_initializer', 'bias_initializer'),
        config.keys())


if __name__ == '__main__':
  tf.test.main()
