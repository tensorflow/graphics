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
"""Tests for safe_ops."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import test_case


def _pick_random_vector():
  """Creates a random vector with random shape."""
  tensor_size = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
  return np.random.normal(size=tensor_shape + [4])


class SafeOpsTest(test_case.TestCase):

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_safe_unsigned_div_exception_not_raised(self, dtype):
    """Checks that unsigned division does not cause Inf values."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)

    self.assert_exception_is_not_raised(
        safe_ops.safe_unsigned_div,
        shapes=[],
        a=tf.norm(tensor=vector),
        b=tf.norm(tensor=zero_vector))

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_safe_unsigned_div_exception_raised(self, dtype):
    """Checks that unsigned division causes Inf values for zero eps."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          safe_ops.safe_unsigned_div(
              tf.norm(tensor=vector), tf.norm(tensor=zero_vector), eps=0.0))

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_safe_signed_div_exception_not_raised(self, dtype):
    """Checks that signed division does not cause Inf values."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)

    self.assert_exception_is_not_raised(
        safe_ops.safe_signed_div,
        shapes=[],
        a=tf.norm(tensor=vector),
        b=tf.sin(zero_vector))

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_safe_signed_div_exception_raised(self, dtype):
    """Checks that signed division causes Inf values for zero eps."""
    vector = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    zero_vector = tf.zeros_like(vector)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          safe_ops.safe_unsigned_div(
              tf.norm(tensor=vector), tf.sin(zero_vector), eps=0.0))

  @parameterized.parameters(tf.float32, tf.float64)
  def test_safe_shrink_exception_not_raised(self, dtype):
    """Checks whether safe shrinking makes tensor safe for tf.acos(x)."""
    tensor = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    tensor = tensor * tensor
    norm_tensor = tensor / tf.reduce_max(
        input_tensor=tensor, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    norm_tensor += eps

    safe_tensor = safe_ops.safe_shrink(norm_tensor, -1.0, 1.0)
    self.assert_exception_is_not_raised(tf.acos, shapes=[], x=safe_tensor)

  @parameterized.parameters(tf.float32, tf.float64)
  def test_safe_shrink_exception_raised(self, dtype):
    """Checks whether safe shrinking fails when eps is zero."""
    tensor = tf.convert_to_tensor(value=_pick_random_vector(), dtype=dtype)
    tensor = tensor * tensor
    norm_tensor = tensor / tf.reduce_max(
        input_tensor=tensor, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    norm_tensor += eps

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(safe_ops.safe_shrink(norm_tensor, -1.0, 1.0, eps=0.0))

  def test_safe_sinpx_div_sinx(self):
    """Tests for edge cases and continuity for sin(px)/sin(x)."""
    angle_step = np.pi / 16.0

    with self.subTest(name="all_angles"):
      theta = tf.range(-2.0 * np.pi, 2.0 * np.pi + angle_step / 2.0, angle_step)
      factor = np.random.uniform(size=(1,))

      division = safe_ops.safe_sinpx_div_sinx(theta, factor)
      division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, factor)
      division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, rtol=1e-9)
      self.assertAllClose(division, division_r, rtol=1e-9)

    with self.subTest(name="theta_is_zero"):
      theta = 0.0
      factor = tf.range(0.0, 1.0, 0.001)

      division = safe_ops.safe_sinpx_div_sinx(theta, factor)
      division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, factor)
      division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, atol=1e-9)
      self.assertAllClose(division, division_r, atol=1e-9)
      # According to l'Hopital rule, limit should be factor
      self.assertAllClose(division, factor, atol=1e-9)

    with self.subTest(name="theta_is_pi"):
      theta = np.pi
      factor = tf.range(0.0, 1.001, 0.001)

      division = safe_ops.safe_sinpx_div_sinx(theta, factor)
      division_l = safe_ops.safe_sinpx_div_sinx(theta + 1e-10, factor)
      division_r = safe_ops.safe_sinpx_div_sinx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, atol=1e-9)
      self.assertAllClose(division, division_r, atol=1e-9)

  def test_safe_cospx_div_cosx(self):
    """Tests for edge cases and continuity for cos(px)/cos(x)."""
    angle_step = np.pi / 16.0

    with self.subTest(name="all_angles"):
      theta = tf.range(-2.0 * np.pi, 2.0 * np.pi + angle_step / 2.0, angle_step)
      factor = np.random.uniform(size=(1,))

      division = safe_ops.safe_cospx_div_cosx(theta, factor)
      division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, factor)
      division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, rtol=1e-9)
      self.assertAllClose(division, division_r, rtol=1e-9)

    with self.subTest(name="theta_is_pi_over_two"):
      theta = np.pi / 2.0
      factor = tf.constant(1.0)

      division = safe_ops.safe_cospx_div_cosx(theta, factor)
      division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, factor)
      division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, atol=1e-9)
      self.assertAllClose(division, division_r, atol=1e-9)
      # According to l'Hopital rule, limit should be 1.0
      self.assertAllClose(division, 1.0, atol=1e-9)

    with self.subTest(name="theta_is_three_pi_over_two"):
      theta = np.pi * 3.0 / 2.0
      factor = tf.range(0.0, 1.001, 0.001)

      division = safe_ops.safe_cospx_div_cosx(theta, factor)
      division_l = safe_ops.safe_cospx_div_cosx(theta + 1e-10, factor)
      division_r = safe_ops.safe_cospx_div_cosx(theta - 1e-10, factor)

      self.assertAllClose(division, division_l, atol=1e-9)
      self.assertAllClose(division, division_r, atol=1e-9)


if __name__ == "__main__":
  test_case.main()
