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
"""Tests for shape utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.util import test_case


class TestCaseTest(test_case.TestCase):

  def _dummy_tf_lite_compatible_function(self, data):
    """Executes a simple supported function to test TFLite conversion."""
    data = tf.convert_to_tensor(value=data)
    return 2.0 * data

  def _dummy_tf_lite_incompatible_function(self, data):
    """Executes a simple unsupported function to test TFLite conversion."""
    del data  # Unused
    return 2.0 * tf.ones(shape=[2] * 10)

  @parameterized.parameters(None, (((1.0,),),))
  def test_assert_tf_lite_convertible_exception_not_raised(self, test_inputs):
    """Tests that assert_tf_lite_convertible succeeds with a simple function."""
    tc = test_case.TestCase(methodName="assert_tf_lite_convertible")

    # We can't use self.assert_exception_is_not_raised here because we need to
    # use `shapes` as both a named argument and a kwarg.
    try:
      tc.assert_tf_lite_convertible(
          func=self._dummy_tf_lite_compatible_function,
          shapes=((1,),),
          test_inputs=test_inputs)
    except unittest.SkipTest as e:
      # Forwarding SkipTest exception in order to skip the test.
      raise e
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % type(e))

  @parameterized.parameters(None, (((1.0,),),))
  def test_assert_tf_lite_convertible_exception_raised(self, test_inputs):
    """Tests that assert_tf_lite_convertible succeeds with a simple function."""
    # TODO(b/131912561): TFLite conversion throws SIGABRT instead of Exception.
    return
    # pylint: disable=unreachable
    # This code should be able to catch exceptions correctly once TFLite bug
    # is fixed.
    tc = test_case.TestCase(methodName="assert_tf_lite_convertible")

    with self.assertRaises(Exception):
      tc.assert_tf_lite_convertible(
          func=self._dummy_tf_lite_incompatible_function,
          shapes=((1,),),
          test_inputs=test_inputs)
    # pylint: enable=unreachable

  def _dummy_failing_function(self, data):
    """Fails instantly."""
    del data  # Unused
    raise ValueError("Fail.")

  def test_assert_exception_is_not_raised_raises_exception(self):
    """Tests that assert_exception_is_not_raised raises exception."""
    if tf.executing_eagerly():
      # In eager mode placeholders are assigned zeros by default, which fails
      # for various tests. Therefore this function can only be tested in graph
      # mode.
      return
    tc = test_case.TestCase(methodName="assert_exception_is_not_raised")

    with self.assertRaises(AssertionError):
      tc.assert_exception_is_not_raised(
          self._dummy_failing_function, shapes=((1,),))


if __name__ == "__main__":
  test_case.main()
