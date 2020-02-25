#Copyright 2019 Google LLC
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
"""Tests for asserts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import test_case


def _pick_random_vector():
  """Creates a random vector with a random shape."""
  tensor_size = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
  return np.random.normal(size=tensor_shape + [4])


class AssertsTest(test_case.TestCase):

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_normalized_exception_not_raised(self, dtype):
    """Checks that assert_normalized raises no exceptions for valid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)
    norm_vector = vector / tf.norm(tensor=vector, axis=-1, keepdims=True)

    self.assert_exception_is_not_raised(
        asserts.assert_normalized, shapes=[], vector=norm_vector)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_normalized_exception_raised(self, dtype):
    """Checks that assert_normalized raises exceptions for invalid input."""
    vector = _pick_random_vector() + 10.0
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)
    vector = tf.abs(vector)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(asserts.assert_normalized(vector))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_normalized_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_normalized(vector_input)

    self.assertIs(vector_input, vector_output)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_at_least_k_non_zero_entries_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_at_least_k_non_zero_entries(vector_input)

    self.assertIs(vector_input, vector_output)

  @parameterized.parameters(
      (None, None),
      (1e-3, tf.float16),
      (4e-19, tf.float32),
      (4e-154, tf.float64),
  )
  def test_assert_nonzero_norm_exception_not_raised(self, value, dtype):
    """Checks that assert_nonzero_norm works for values above eps."""
    if value is None:
      vector = _pick_random_vector() + 10.0
      vector = tf.convert_to_tensor(value=vector, dtype=dtype)
      vector = tf.abs(vector)
    else:
      vector = tf.constant((value,), dtype=dtype)

    self.assert_exception_is_not_raised(
        asserts.assert_nonzero_norm, shapes=[], vector=vector)

  @parameterized.parameters(
      (1e-4, tf.float16),
      (1e-38, tf.float32),
      (1e-308, tf.float64),
  )
  def test_assert_nonzero_norm_exception_raised(self, value, dtype):
    """Checks that assert_nonzero_norm fails for values below eps."""
    vector = tf.constant((value,), dtype=dtype)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(asserts.assert_nonzero_norm(vector))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_nonzero_norm_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_nonzero_norm(vector_input)

    self.assertIs(vector_input, vector_output)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_above_exception_not_raised(self, dtype):
    """Checks that assert_all_above raises no exceptions for valid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= -tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector + eps
    ones_vector = -tf.ones_like(vector)

    with self.subTest(name="inside_and_open_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_above,
          shapes=[],
          vector=inside_vector,
          minval=-1.0,
          open_bound=True)

    with self.subTest(name="inside_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_above,
          shapes=[],
          vector=inside_vector,
          minval=-1.0,
          open_bound=False)

    with self.subTest(name="exact_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_above,
          shapes=[],
          vector=ones_vector,
          minval=-1.0,
          open_bound=False)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_above_exception_raised(self, dtype):
    """Checks that assert_all_above raises exceptions for invalid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= -tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    outside_vector = vector - eps
    ones_vector = -tf.ones_like(vector)

    with self.subTest(name="outside_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_above(outside_vector, -1.0, open_bound=True))

    with self.subTest(name="outside_and_close_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_above(outside_vector, -1.0, open_bound=False))

    with self.subTest(name="exact_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_above(ones_vector, -1.0, open_bound=True))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_above_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_all_above(vector_input, 1.0)

    self.assertIs(vector_input, vector_output)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_below_exception_not_raised(self, dtype):
    """Checks that assert_all_below raises no exceptions for valid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector - eps
    ones_vector = tf.ones_like(vector)

    with self.subTest(name="inside_and_open_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_below,
          shapes=[],
          vector=inside_vector,
          maxval=1.0,
          open_bound=True)

    with self.subTest(name="inside_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_below,
          shapes=[],
          vector=inside_vector,
          maxval=1.0,
          open_bound=False)

    with self.subTest(name="exact_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_below,
          shapes=[],
          vector=ones_vector,
          maxval=1.0,
          open_bound=False)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_below_exception_raised(self, dtype):
    """Checks that assert_all_below raises exceptions for invalid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    outside_vector = vector + eps
    ones_vector = tf.ones_like(vector)

    with self.subTest(name="outside_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_below(outside_vector, 1.0, open_bound=True))

    with self.subTest(name="outside_and_close_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_below(outside_vector, 1.0, open_bound=False))

    with self.subTest(name="exact_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_below(ones_vector, 1.0, open_bound=True))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_below_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()
    vector_output = asserts.assert_all_below(vector_input, 0.0)

    self.assertIs(vector_input, vector_output)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_in_range_exception_not_raised(self, dtype):
    """Checks that assert_all_in_range raises no exceptions for valid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    inside_vector = vector - eps
    ones_vector = tf.ones_like(vector)

    with self.subTest(name="inside_and_open_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_in_range,
          shapes=[],
          vector=inside_vector,
          minval=-1.0,
          maxval=1.0,
          open_bounds=True)

    with self.subTest(name="inside_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_in_range,
          shapes=[],
          vector=inside_vector,
          minval=-1.0,
          maxval=1.0,
          open_bounds=False)

    with self.subTest(name="exact_and_close_bounds"):
      self.assert_exception_is_not_raised(
          asserts.assert_all_in_range,
          shapes=[],
          vector=ones_vector,
          minval=-1.0,
          maxval=1.0,
          open_bounds=False)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_assert_all_in_range_exception_raised(self, dtype):
    """Checks that assert_all_in_range raises exceptions for invalid input."""
    vector = _pick_random_vector()
    vector = tf.convert_to_tensor(value=vector, dtype=dtype)

    vector = vector * vector
    vector /= tf.reduce_max(input_tensor=vector, axis=-1, keepdims=True)
    eps = asserts.select_eps_for_addition(dtype)
    outside_vector = vector + eps
    ones_vector = tf.ones_like(vector)

    with self.subTest(name="outside_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_in_range(
                outside_vector, -1.0, 1.0, open_bounds=True))

    with self.subTest(name="outside_and_close_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_in_range(
                outside_vector, -1.0, 1.0, open_bounds=False))

    with self.subTest(name="exact_and_open_bounds"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            asserts.assert_all_in_range(
                ones_vector, -1.0, 1.0, open_bounds=True))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_all_in_range_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_all_in_range(vector_input, -1.0, 1.0)

    self.assertIs(vector_input, vector_output)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_select_eps_for_division(self, dtype):
    """Checks that select_eps_for_division does not cause Inf values."""
    a = tf.constant(1.0, dtype=dtype)
    eps = asserts.select_eps_for_division(dtype)

    self.assert_exception_is_not_raised(
        asserts.assert_no_infs_or_nans, shapes=[], tensor=a / eps)

  @parameterized.parameters(tf.float16, tf.float32, tf.float64)
  def test_select_eps_for_addition(self, dtype):
    """Checks that select_eps_for_addition returns large enough eps."""
    a = tf.constant(1.0, dtype=dtype)
    eps = asserts.select_eps_for_addition(dtype)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(tf.compat.v1.assert_equal(a, a + eps))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_no_infs_or_nans_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    vector_input = _pick_random_vector()

    vector_output = asserts.assert_no_infs_or_nans(vector_input)

    self.assertIs(vector_input, vector_output)


if __name__ == "__main__":
  test_case.main()
