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
"""Tests for tangential_distortion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_graphics.rendering.camera import tangential_distortion
from tensorflow_graphics.util import test_case

RANDOM_TESTS_NUM_IMAGES = 10
RANDOM_TESTS_HEIGHT = 8
RANDOM_TESTS_WIDTH = 8

RADII_SHAPE = (RANDOM_TESTS_NUM_IMAGES, RANDOM_TESTS_HEIGHT, RANDOM_TESTS_WIDTH)
COORDINATES_SHAPE = (RANDOM_TESTS_NUM_IMAGES, RANDOM_TESTS_HEIGHT, RANDOM_TESTS_WIDTH)
COEFFICIENT_SHAPE = (RANDOM_TESTS_NUM_IMAGES,)


def _get_random_coordinates():
  return np.ones(shape=COORDINATES_SHAPE).astype('float32')


def _get_zeros_coordinates():
  return np.zeros(shape=COORDINATES_SHAPE).astype('float32')


def _get_ones_coordinates():
  return np.ones(shape=COORDINATES_SHAPE).astype('float32')


def _get_random_coefficient():
  return np.random.rand(*COEFFICIENT_SHAPE).astype('float32')


def _get_zeros_coefficient():
  return np.zeros(shape=COEFFICIENT_SHAPE).astype('float32')


def _get_ones_coefficient():
  return np.ones(shape=COEFFICIENT_SHAPE).astype('float32')


def _make_shape_compatible(coefficients):
  return np.expand_dims(np.expand_dims(coefficients, axis=-1), axis=-1)

def _get_squared_radii(projective_x, projective_y):
  return projective_x ** 2.0 + projective_y ** 2.0

class TangentialDistortionTest(test_case.TestCase):

  def test_distortion_terms_random_positive_distortion_coefficients(self):
    """Tests that distortion_terms produces the expected outputs."""
    projective_x = _get_random_coordinates() * 2.0
    projective_y = _get_random_coordinates() * 2.0
    distortion_coefficient_1 = _get_random_coefficient() * 2.0
    distortion_coefficient_2 = _get_random_coefficient() * 2.0

    distortion_x, distortion_y = tangential_distortion.distortion_terms(
      projective_x, projective_y, distortion_coefficient_1,
      distortion_coefficient_2)

    distortion_coefficient_1 = _make_shape_compatible(distortion_coefficient_1)
    distortion_coefficient_2 = _make_shape_compatible(distortion_coefficient_2)
    squared_radius = _get_squared_radii(projective_x, projective_y)
    with self.subTest(name='distortion'):
      self.assertAllClose(2.0 * distortion_coefficient_1 * projective_x *
                          projective_y + distortion_coefficient_2 *
                          (squared_radius + 2 * projective_x ** 2.0),
                          distortion_x)
      self.assertAllClose(2.0 * distortion_coefficient_2 * projective_x *
                          projective_y + distortion_coefficient_1 *
                          (squared_radius + 2 * projective_y ** 2.0),
                          distortion_y)

    def test_distortion_terms_preset_zero_distortion_coefficients(self):
      """Tests distortion_terms at zero disortion coefficients."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0

      distortion_x, distortion_y = tangential_distortion.distortion_terms(
        projective_x, projective_y, 0.0, 0.0)

      with self.subTest(name='distortion'):
        self.assertAllClose(tf.zeros_like(projective_x), distortion_x)
        self.assertAllClose(tf.zeros_like(projective_y), distortion_y)


if __name__ == '__main__':
  test_case.main()
