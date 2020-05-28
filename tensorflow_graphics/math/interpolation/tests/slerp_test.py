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
"""Tests for slerp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math.interpolation import slerp
from tensorflow_graphics.util import test_case

_SQRT2_DIV2 = np.sqrt(2.0).astype(np.float32) * 0.5


class SlerpTest(test_case.TestCase):

  def _pick_random_quaternion(self):
    """Creates a random quaternion with random shape."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    return np.random.normal(size=tensor_shape + [4])

  def _quaternion_slerp_helper(self, q1, q2, p):
    """Calls interpolate function for quaternions."""
    return slerp.interpolate(q1, q2, p, slerp.InterpolationType.QUATERNION)

  def _vector_slerp_helper(self, q1, q2, p):
    """Calls interpolate function for vectors."""
    return slerp.interpolate(q1, q2, p, slerp.InterpolationType.VECTOR)

  def test_interpolate_raises_exceptions(self):
    """Tests if unknown methods raise exceptions."""
    vector1 = self._pick_random_quaternion()
    self.assert_exception_is_raised(
        slerp.interpolate,
        error_msg="Unknown interpolation type supplied.",
        shapes=[],
        vector1=vector1,
        vector2=-vector1,
        percent=0.1,
        method=2)

  def test_interpolate_with_weights_quaternion_preset(self):
    """Compares interpolate to quaternion_weights + interpolate_with_weights."""
    q1 = self._pick_random_quaternion()
    q2 = q1 + tf.ones_like(q1)
    q1 = tf.nn.l2_normalize(q1, axis=-1)
    q2 = tf.nn.l2_normalize(q2, axis=-1)

    weight1, weight2 = slerp.quaternion_weights(q1, q2, 0.25)
    qf = slerp.interpolate_with_weights(q1, q2, weight1, weight2)
    qi = slerp.interpolate(
        q1, q2, 0.25, method=slerp.InterpolationType.QUATERNION)

    self.assertAllClose(qf, qi, atol=1e-9)

  def test_interpolate_with_weights_vector_preset(self):
    """Compares interpolate to vector_weights + interpolate_with_weights."""
    # Any quaternion is a valid vector
    q1 = self._pick_random_quaternion()
    q2 = q1 + tf.ones_like(q1)

    weight1, weight2 = slerp.vector_weights(q1, q2, 0.75)
    qf = slerp.interpolate_with_weights(q1, q2, weight1, weight2)

    qi = slerp.interpolate(q1, q2, 0.75, method=slerp.InterpolationType.VECTOR)
    self.assertAllClose(qf, qi, atol=1e-9)

  @parameterized.parameters(
      # Orthogonal, same hemisphere
      (((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.5,)),
       ((_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0),)),
      (((_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0),
        (0.0, 0.0, _SQRT2_DIV2, _SQRT2_DIV2), (0.5,)), ((0.5, 0.5, 0.5, 0.5),)),
      # Same hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (0.0, 0.0, _SQRT2_DIV2, _SQRT2_DIV2), (0.5,)),
       ((0.408248290463863, 0.0, 0.816496580927726, 0.408248290463863),)),
      # Same quaternions
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0), (0.75,)),
       ((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),)),
      # Anti-polar - small percent
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.2,)),
       ((-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0),)),
      # Anti-polar - large percent
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.8,)),
       ((-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0),)),
      # Extrapolation - same hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0), (-0.5,)),
       ((0.408248290463863, -0.408248290463863, 0.816496580927726, 0.0),)),
      # Extrapolation - opposite hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0), (-0.5,)),
       ((-0.408248290463863, -0.408248290463863, -0.816496580927726, 0.0),)),
  )
  def test_quaternion_slerp_preset(self, test_inputs, test_outputs):
    """Tests the accuracy of qslerp against numpy-quaternion values."""
    test_inputs = [np.array(test_input).astype(np.float32)
                   for test_input in test_inputs]
    self.assert_output_is_correct(self._quaternion_slerp_helper, test_inputs,
                                  test_outputs, tile=False)

  def test_unnormalized_quaternion_weights_exception_raised(self):
    """Tests if quaternion_weights raise exceptions for unnormalized input."""
    q1 = self._pick_random_quaternion()
    q2 = tf.nn.l2_normalize(q1, axis=-1)
    p = tf.constant((0.5), dtype=q1.dtype)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(slerp.quaternion_weights(q1, q2, p))

  @parameterized.parameters(
      ((4,), (4,), (1,)),
      ((None, 4), (None, 4), (None, 1)),
      ((None, 4), (None, 4), (None, 4)),
  )
  def test_quaternion_weights_exception_not_raised(self, *shapes):
    """Tests that valid input shapes do not raise exceptions for qslerp."""
    self.assert_exception_is_not_raised(slerp.quaternion_weights, shapes)

  @parameterized.parameters(
      ("must have exactly 4 dimensions in axis -1", (3,), (4,), (1,)),
      ("must have exactly 4 dimensions in axis -1", (4,), (3,), (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 4), (3, 4),
       (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 4), (3, 4),
       (2,)),
  )
  def test_quaternion_weights_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised for qslerp."""
    self.assert_exception_is_raised(slerp.quaternion_weights, error_msg, shapes)

  @parameterized.parameters(
      # Same quaternions
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0), (0.75,)), (
            (0.25,),
            (0.75,),
        )),
      # Anti-polar - small percent
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.2,)), (
            (-0.8,),
            (0.2,),
        )),
      # Anti-polar - large percent
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.8,)), (
            (-0.2,),
            (0.8,),
        )),
  )
  def test_quaternion_weights_preset(self, test_inputs, test_outputs):
    """Tests the accuracy of quaternion_weights for problem cases."""
    test_inputs = [np.array(test_input).astype(np.float32)
                   for test_input in test_inputs]
    self.assert_output_is_correct(slerp.quaternion_weights, test_inputs,
                                  test_outputs, tile=False)

  @parameterized.parameters(
      ((3,), (3,), (1,)),
      ((None, 4), (None, 4), (None, 1)),
  )
  def test_vector_weights_exception_not_raised(self, *shapes):
    """Tests that valid inputs do not raise exceptions for vector_weights."""
    self.assert_exception_is_not_raised(slerp.vector_weights, shapes)

  @parameterized.parameters(
      ("must have the same number of dimensions in axes", (None, 3), (None, 4),
       (1,)),
      ("must have the same number of dimensions in axes", (2, 3), (2, 4), (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3),
       (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 3), (3, 3),
       (2,)),
  )
  def test_vector_weights_exception_raised(self, error_msg, *shapes):
    """Tests that shape exceptions are properly raised for vector_weights."""
    self.assert_exception_is_raised(slerp.vector_weights, error_msg, shapes)

  @parameterized.parameters(
      # Orthogonal, same hemisphere
      (((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.5,)),
       ((_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0),)),
      (((_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0),
        (0.0, 0.0, _SQRT2_DIV2, _SQRT2_DIV2), (0.5,)), ((0.5, 0.5, 0.5, 0.5),)),
      # Same hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (0.0, 0.0, _SQRT2_DIV2, _SQRT2_DIV2), (0.5,)),
       ((0.408248290463863, 0.0, 0.816496580927726, 0.408248290463863),)),
      # Same vectors
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0), (0.75,)),
       ((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),)),
      # Anti-polar - equal weights
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.5,)),
       ((0.0, 0.0, 0.0, 0.0),)),
      # Anti-polar - small percent
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, 0.0, -_SQRT2_DIV2, 0.0), (0.25,)),
       ((0.5, 0.0, 0.5, 0.0),)),
      # Extrapolation - same hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0), (-1.0,)),
       ((0.0, -_SQRT2_DIV2, _SQRT2_DIV2, 0.0),)),
      # Extrapolation - opposite hemisphere
      (((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0),
        (-_SQRT2_DIV2, _SQRT2_DIV2, 0.0, 0.0), (1.5,)),
       ((-_SQRT2_DIV2, -0.0, -_SQRT2_DIV2, 0.0),)),
      # Unnormalized vectors
      (((4.0, 0.0), (0.0, 1.0), (0.5,)), ((2.82842712, _SQRT2_DIV2),)),
  )
  def test_vector_slerp_preset(self, test_inputs, test_outputs):
    """Tests the accuracy of vector slerp results."""
    test_inputs = [np.array(test_input).astype(np.float32)
                   for test_input in test_inputs]
    self.assert_output_is_correct(self._vector_slerp_helper, test_inputs,
                                  test_outputs, tile=False)

  def test_vector_weights_reduce_to_lerp_preset(self):
    """Tests if vector slerp reduces to lerp for identical vectors as input."""
    q1 = tf.constant((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0))
    q2 = tf.constant((_SQRT2_DIV2, 0.0, _SQRT2_DIV2, 0.0))
    p = tf.constant((0.75,), dtype=q1.dtype)

    w1, w2 = slerp.vector_weights(q1, q2, p)

    self.assertAllClose(w1, (0.25,), rtol=1e-6)
    self.assertAllClose(w2, (0.75,), rtol=1e-6)


if __name__ == "__main__":
  test_case.main()
