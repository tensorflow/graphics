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
"""Unit test base class.

This class is intended to be used as the unit test base class in TensorFlow
Graphics. It implements new methods on top of the TensorFlow TestCase class
that are used to simplify the code and check for various kinds of failure.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import tfg_flags

FLAGS = flags.FLAGS


def _max_error(arrays1, arrays2):
  """Computes maximum elementwise gap between two lists of ndarrays.

  Computes the maximum elementwise gap between two lists with the same length,
  of arrays with the same shape.

  Args:
    arrays1: a lists of np.ndarrays.
    arrays2: a lists of np.ndarrays of the same shape as arrays1.

  Returns:
    The maximum elementwise absolute difference between the two lists of arrays.
  """
  error = 0
  for array1, array2 in zip(arrays1, arrays2):
    if array1.size or array2.size:  # Handle zero size ndarrays correctly
      error = np.maximum(error, np.fabs(array1 - array2).max())
  return error


class TestCase(parameterized.TestCase, tf.test.TestCase):
  """Test case class implementing extra test functionalities."""

  def setUp(self):  # pylint: disable=invalid-name
    """Sets the seed for tensorflow and numpy."""
    super(TestCase, self).setUp()
    try:
      seed = flags.FLAGS.test_random_seed
    except flags.UnparsedFlagAccessError:
      seed = 301  # Default seed in case test_random_seed is not defined.
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True

  def _remove_dynamic_shapes(self, shapes):
    for s in shapes:
      if None in s:
        return None
    return shapes

  def _compute_gradient_error(self, x, y, x_init_value, delta=1e-6):
    """Computes the gradient error.

    Args:
      x: a tensor or list of tensors.
      y: a tensor.
      x_init_value: a numpy array of the same shape as "x" representing the
        initial value of x.
      delta: (optional) the amount of perturbation.

    Returns:
      A tuple (max_error, row, column), with max_error the maxium error between
      the two Jacobians, and row/column the position of said maximum error.
    """
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()
    with self.cached_session():
      grad = tf.compat.v1.test.compute_gradient(x, x_shape, y, y_shape,
                                                x_init_value, delta)
      if isinstance(grad, tuple):
        grad = [grad]
      error = 0
      row_max_error = 0
      column_max_error = 0
      for j_t, j_n in grad:
        if j_t.size or j_n.size:  # Handle zero size tensors correctly
          diff = np.fabs(j_t - j_n)
          max_error = np.maximum(error, diff.max())
          row_max_error, column_max_error = np.unravel_index(
              diff.argmax(), diff.shape)
      return max_error, row_max_error, column_max_error

  def _create_placeholders_from_shapes(self,
                                       shapes,
                                       dtypes=None,
                                       sparse_tensors=None):
    """Creates a list of placeholders based on a list of shapes.

    Args:
      shapes: A tuple or list of the input shapes.
      dtypes: A list of input types.
      sparse_tensors: A `bool` list denoting if placeholder is a SparseTensor.
        This is ignored in eager mode - in eager execution, only dense
        placeholders will be created.

    Returns:
      A list of placeholders.
    """
    if dtypes is None:
      dtypes = [tf.float32] * len(shapes)
    if sparse_tensors is None:
      sparse_tensors = [False] * len(shapes)
    if tf.executing_eagerly():
      placeholders = [
          tf.compat.v1.placeholder_with_default(
              tf.zeros(shape=shape, dtype=dtype), shape=shape)
          for shape, dtype in zip(shapes, dtypes)
      ]
    else:
      placeholders = [
          tf.compat.v1.sparse.placeholder(dtype, shape=shape)
          if is_sparse else tf.compat.v1.placeholder(shape=shape, dtype=dtype)
          for shape, dtype, is_sparse in zip(shapes, dtypes, sparse_tensors)
      ]
    return placeholders

  def _tile_tensors(self, tiling, tensors):
    """Tiles a set of tensors using the tiling information.

    Args:
      tiling: A list of integers defining how to tile the tensors.
      tensors: A list of tensors to tile.

    Returns:
      A list of tiled tensors.
    """
    tensors = [
        np.tile(tensor, tiling + [1] * len(np.array(tensor).shape))
        for tensor in tensors
    ]
    return tensors

  def assert_exception_is_not_raised(self,
                                     func,
                                     shapes,
                                     dtypes=None,
                                     sparse_tensors=None,
                                     **kwargs):
    """Runs the function to make sure an exception is not raised.

    Args:
      func: A function to exectute.
      shapes: A tuple or list of the input shapes.
      dtypes: A list of input types.
      sparse_tensors: A list of `bool` indicating if the inputs are
        SparseTensors. Defaults to all `False`. This is used for creating
        SparseTensor placeholders in graph mode.
      **kwargs: A dict of keyword arguments to be passed to the function.
    """
    if tf.executing_eagerly() and shapes:
      # If a shape is given in eager mode, the tensor will be initialized with
      # zeros, which can make some range checks fail for certain functions.
      # But if only kwargs are passed and shapes is empty, this function
      # still should run correctly.
      return
    placeholders = self._create_placeholders_from_shapes(
        shapes=shapes, dtypes=dtypes, sparse_tensors=sparse_tensors)
    try:
      func(*placeholders, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def assert_exception_is_raised(self,
                                 func,
                                 error_msg,
                                 shapes,
                                 dtypes=None,
                                 sparse_tensors=None,
                                 **kwargs):
    """Runs the function to make sure an exception is raised.

    Args:
      func: A function to exectute.
      error_msg: The error message of the exception.
      shapes: A tuple or list of the input shapes.
      dtypes: A list of input types.
      sparse_tensors: A list of `bool` indicating if the inputs are
        SparseTensors. Defaults to all `False`. This is used for creating
        SparseTensor placeholders in graph mode.
      **kwargs: A dict of keyword arguments to be passed to the function.
    """
    if tf.executing_eagerly():
      # If shapes is an empty list, we can continue with the test. If shapes
      # has None values, we shoud return.
      shapes = self._remove_dynamic_shapes(shapes)
      if shapes is None:
        return
    placeholders = self._create_placeholders_from_shapes(
        shapes=shapes, dtypes=dtypes, sparse_tensors=sparse_tensors)
    with self.assertRaisesRegexp(ValueError, error_msg):
      func(*placeholders, **kwargs)

  def assert_jacobian_is_correct(self, x, x_init, y, atol=1e-6, delta=1e-6):
    """Tests that the gradient error of y=f(x) is small.

    Args:
      x: A tensor.
      x_init: A numpy array containing the values at which to estimate the
        gradients of y.
      y: A tensor.
      atol: Maximum absolute tolerance in gradient error.
      delta: The amount of perturbation.
    """
    warnings.warn((
        "assert_jacobian_is_correct is deprecated and might get "
        "removed in a future version please use assert_jacobian_is_correct_fn"),
                  DeprecationWarning)
    if tf.executing_eagerly():
      self.skipTest(reason="Graph mode only test")
    max_error, _, _ = self._compute_gradient_error(x, y, x_init, delta)
    self.assertLessEqual(max_error, atol)

  def assert_jacobian_is_correct_fn(self, f, x, atol=1e-6, delta=1e-6):
    """Tests that the gradient error of y=f(x) is small.

    Args:
      f: the function.
      x: A list of arguments for the function
      atol: Maximum absolute tolerance in gradient error.
      delta: The amount of perturbation.
    """
    # pylint: disable=no-value-for-parameter
    if tf.executing_eagerly():
      max_error = _max_error(*tf.test.compute_gradient(f, x, delta))
    else:
      with self.cached_session():
        max_error = _max_error(*tf.test.compute_gradient(f, x, delta))
    # pylint: enable=no-value-for-parameter
    self.assertLessEqual(max_error, atol)

  def assert_jacobian_is_finite(self, x, x_init, y):
    """Tests that the Jacobian only contains valid values.

    The analytical gradients and numerical ones are expected to differ at points
    where y is not smooth. This function can be used to check that the
    analytical gradient is not NaN nor Inf.

    Args:
      x: A tensor.
      x_init: A numpy array containing the values at which to estimate the
        gradients of y.
      y: A tensor.
    """
    warnings.warn((
        "assert_jacobian_is_finite is deprecated and might get "
        "removed in a future version please use assert_jacobian_is_finite_fn"),
                  DeprecationWarning)
    if tf.executing_eagerly():
      self.skipTest(reason="Graph mode only test")
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()
    with tf.compat.v1.Session():
      gradient = tf.compat.v1.test.compute_gradient(
          x, x_shape, y, y_shape, x_init_value=x_init)
      theoretical_gradient = gradient[0][0]
      self.assertFalse(
          np.isnan(theoretical_gradient).any() or
          np.isinf(theoretical_gradient).any())

  def assert_jacobian_is_finite_fn(self, f, x):
    """Tests that the Jacobian only contains valid values.

    The analytical gradients and numerical ones are expected to differ at points
    where f(x) is not smooth. This function can be used to check that the
    analytical gradient is not 'NaN' nor 'Inf'.

    Args:
      f: the function.
      x: A list of arguments for the function
    """
    if tf.executing_eagerly():
      theoretical_gradient, _ = tf.compat.v2.test.compute_gradient(f, x)
    else:
      with self.cached_session():
        theoretical_gradient, _ = tf.compat.v2.test.compute_gradient(f, x)
    self.assertNotIn(
        True, [
            np.isnan(element).any() or np.isinf(element).any()
            for element in theoretical_gradient
        ],
        msg="nan or inf elements found in theoretical jacobian.")

  def assert_output_is_correct(self,
                               func,
                               test_inputs,
                               test_outputs,
                               rtol=1e-3,
                               atol=1e-6,
                               tile=True):
    """Tests that the function gives the correct result.

    Args:
      func: A function to exectute.
      test_inputs: A tuple or list of test inputs.
      test_outputs: A tuple or list of test outputs against which the result of
        calling `func` on `test_inputs` will be compared to.
      rtol: The relative tolerance used during the comparison.
      atol: The absolute tolerance used during the comparison.
      tile: A `bool` indicating whether or not to automatically tile the test
        inputs and outputs.
    """
    if tile:
      # Creates a rank 4 list of values between 1 and 10.
      tensor_tile = np.random.randint(1, 10, size=np.random.randint(4)).tolist()
      test_inputs = self._tile_tensors(tensor_tile, test_inputs)
      test_outputs = self._tile_tensors(tensor_tile, test_outputs)
    test_outputs = [
        tf.convert_to_tensor(value=output) for output in test_outputs
    ]
    test_outputs = test_outputs[0] if len(test_outputs) == 1 else test_outputs
    self.assertAllClose(test_outputs, func(*test_inputs), rtol=rtol, atol=atol)

  def assert_tf_lite_convertible(self,
                                 func,
                                 shapes,
                                 dtypes=None,
                                 test_inputs=None):
    """Runs the tf-lite converter to make sure the function can be exported.

    Args:
      func: A function to execute with tf-lite.
      shapes: A tuple or list of input shapes.
      dtypes: A list of input types.
      test_inputs: A tuple or list of inputs. If not provided the test inputs
        will be randomly generated.
    """
    if tf.executing_eagerly():
      # Currently TFLite conversion is not supported in eager mode.
      self.skipTest(reason="Graph mode only test")
    # Generate graph with the function given as input.
    in_tensors = self._create_placeholders_from_shapes(shapes, dtypes)
    out_tensors = func(*in_tensors)
    if not isinstance(out_tensors, (list, tuple)):
      out_tensors = [out_tensors]
    with tf.compat.v1.Session() as sess:
      try:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Convert to a TFLite model.
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(
            sess, in_tensors, out_tensors)
        tflite_model = converter.convert()
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        # If no test inputs provided then randomly generate inputs.
        if test_inputs is None:
          test_inputs = [
              np.array(np.random.sample(shape), dtype=np.float32)
              for shape in shapes
          ]
        else:
          test_inputs = [
              np.array(test, dtype=np.float32) for test in test_inputs
          ]
        # Evaluate function using TensorFlow.
        feed_dict = dict(zip(in_tensors, test_inputs))
        test_outputs = sess.run(out_tensors, feed_dict)
        # Set tensors for the TFLite model.
        input_details = interpreter.get_input_details()
        for i, test_input in enumerate(test_inputs):
          index = input_details[i]["index"]
          interpreter.set_tensor(index, test_input)
        # Run TFLite model.
        interpreter.invoke()
        # Get tensors from the TFLite model and compare with TensorFlow.
        output_details = interpreter.get_output_details()
        for o, test_output in enumerate(test_outputs):
          index = output_details[o]["index"]
          self.assertAllClose(test_output, interpreter.get_tensor(index))
      except Exception as e:  # pylint: disable=broad-except
        self.fail("Exception raised: %s" % str(e))


def main(argv=None):
  """Main function."""
  tf.test.main(argv)


# The util functions or classes are not exported.
__all__ = []
