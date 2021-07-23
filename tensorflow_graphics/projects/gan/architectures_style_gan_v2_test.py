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
"""Tests for gan.architectures_style_gan_v2."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.projects.gan import architectures_style_gan_v2


class ArchitecturesStyleGanV2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('batch_1', 1, False), ('batch_2', 2, False),
                                  ('normalize_latent_code', 1, True))
  def test_generator_output_size(self, batch_size, normalize_latent_code):
    input_data = np.ones(shape=(batch_size, 8), dtype=np.float32)
    generator, _, _ = architectures_style_gan_v2.create_style_based_generator(
        latent_code_dimension=8,
        upsampling_blocks_num_channels=(8, 8),
        normalize_latent_code=normalize_latent_code)
    expected_size = 16

    output = generator(input_data)

    self.assertSequenceEqual(output.shape,
                             (batch_size, expected_size, expected_size, 3))

  def test_generator_output_size_with_noise_inputs(self):
    upsampling_blocks_num_channels = (8, 8)
    batch_size = 2
    generator, _, _ = architectures_style_gan_v2.create_style_based_generator(
        latent_code_dimension=8,
        upsampling_blocks_num_channels=upsampling_blocks_num_channels,
        use_noise_inputs=True)
    noise_dimensions = architectures_style_gan_v2.get_noise_dimensions(
        num_upsampling_blocks=len(upsampling_blocks_num_channels))
    input_noise = [
        np.zeros((batch_size,) + noise_dimension, dtype=np.float32)
        for noise_dimension in noise_dimensions
    ]
    input_data = np.ones(shape=(batch_size, 8), dtype=np.float32)
    expected_size = 16

    output = generator([input_data] + input_noise)

    self.assertSequenceEqual(output.shape,
                             (batch_size, expected_size, expected_size, 3))

  def test_cloning_generator(self):
    generator, _, _ = architectures_style_gan_v2.create_style_based_generator()

    with tf.keras.utils.custom_object_scope(
        architectures_style_gan_v2.CUSTOM_LAYERS):
      generator_clone = tf.keras.models.clone_model(generator)

    self.assertIsInstance(generator_clone, tf.keras.Model)

  @parameterized.named_parameters(
      ('batch_1', 1, False, False), ('batch_2', 2, False, False),
      ('antialiased_bilinear_downsampling', 1, False, True),
      ('scaled_kernels', 1, True, False))
  def test_discriminator_output_size(self, batch_size,
                                     use_fan_in_scaled_kernels,
                                     antialiased_bilinear_downsampling):
    input_data = np.ones(shape=(batch_size, 16, 16, 3), dtype=np.float32)
    discriminator = architectures_style_gan_v2.create_discriminator(
        use_fan_in_scaled_kernels=use_fan_in_scaled_kernels,
        downsampling_blocks_num_channels=((8, 8), (8, 8)),
        use_antialiased_bilinear_downsampling=antialiased_bilinear_downsampling)

    output = discriminator(input_data)

    self.assertSequenceEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
  tf.test.main()
