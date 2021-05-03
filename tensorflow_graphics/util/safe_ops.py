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
"""Safe divisions and inverse trigonometric functions.

  This module implements safety mechanisms to prevent NaN's and Inf's from
  appearing due to machine precision issues. These safety mechanisms ensure that
  the derivative is unchanged and the sign of the perturbation is unbiased.
  If the debug flag TFG_ADD_ASSERTS_TO_GRAPH is set to True, all affected
  functions also add assertions to the graph to ensure that the fix has worked
  as expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import asserts


def nonzero_sign(x, name='nonzero_sign'):
  """Returns the sign of x with sign(0) defined as 1 instead of 0."""
  with tf.name_scope(name):
    x = tf.convert_to_tensor(value=x)

    one = tf.ones_like(x)
    return tf.where(tf.greater_equal(x, 0.0), one, -one)


def safe_cospx_div_cosx(theta, factor, eps=None, name='safe_cospx_div_cosx'):
  """Calculates cos(factor * theta)/cos(theta) safely.

  The term `cos(factor * theta)/cos(theta)` has periodic edge cases with
  division by zero problems, and also zero / zero, e.g. when factor is equal to
  1.0 and `theta` is `(n + 1/2)pi`. This function adds signed eps to the angles
  in both the nominator and the denominator to ensure safety, and returns the
  correct values in all edge cases.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    theta: A tensor of shape `[A1, ..., An]`, representing angles in radians.
    factor: A `float` or a tensor of shape `[A1, ..., An]`.
    eps: A `float`, used to perturb the angle. If left as `None`, its value is
      automatically determined from the `dtype` of `theta`.
    name: A name for this op. Defaults to 'safe_cospx_div_cosx'.

  Raises:
    InvalidArgumentError: If tf-graphics debug flag is set and division returns
      `NaN` or `Inf` values.

  Returns:
    A tensor of shape `[A1, ..., An]` containing the resulting values.
  """
  with tf.name_scope(name):
    theta = tf.convert_to_tensor(value=theta)
    factor = tf.convert_to_tensor(value=factor, dtype=theta.dtype)
    if eps is None:
      eps = asserts.select_eps_for_division(theta.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=theta.dtype)

    # eps will be multiplied with factor next, which can make it zero.
    # Therefore we multiply eps with min(1/factor, 1e10), which can handle
    # factors as small as 1e-10 correctly, while preventing a division by zero.
    eps *= tf.clip_by_value(1.0 / factor, 1.0, 1e10)
    sign = nonzero_sign(0.5 * np.pi - (theta - 0.5 * np.pi) % np.pi)
    theta += sign * eps
    div = tf.cos(factor * theta) / tf.cos(theta)
    return asserts.assert_no_infs_or_nans(div)


def safe_shrink(vector,
                minval=None,
                maxval=None,
                open_bounds=False,
                eps=None,
                name='safe_shrink'):
  """Shrinks vector by (1.0 - eps) based on its dtype.

  This function shrinks the input vector by a very small amount to ensure that
  it is not outside of expected range because of floating point precision
  of operations, e.g. dot product of a normalized vector with itself can
  be greater than `1.0` by a small amount determined by the `dtype` of the
  vector. This function can be used to shrink it without affecting its
  derivative (unlike tf.clip_by_value) and make it safe for other operations
  like `acos(x)`. If the tf-graphics debug flag is set to `True`, this function
  adds assertions to the graph that explicitly check that the vector is in the
  range `[minval, maxval]` when open_bounds is `False`, or in range `]minval,
  maxval[` when open_bounds is `True`.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    vector: A tensor of shape `[A1, ..., An]`.
    minval: A `float` or a tensor of shape `[A1, ..., An]`, which contains the
      the lower bounds for tensor values after shrinking to test against. This
      is only used when both `minval` and `maxval` are not `None`.
    maxval: A `float` or a tensor of shape `[A1, ..., An]`, which contains the
      the upper bounds for tensor values after shrinking to test against. This
      is only used when both `minval` and `maxval` are not `None`.
    open_bounds: A `bool` indicating whether the assumed range is open or
      closed, only to be used when both `minval` and `maxval` are not `None`.
    eps: A `float` that is used to shrink the `vector`. If left as `None`, its
      value is automatically determined from the `dtype` of `vector`.
    name: A name for this op. Defaults to 'safe_shrink'.

  Raises:
    InvalidArgumentError: If tf-graphics debug flag is set and the vector is not
      inside the expected range.

  Returns:
    A tensor of shape `[A1, ..., An]` containing the shrinked values.
  """
  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    if eps is None:
      eps = asserts.select_eps_for_addition(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

    vector *= (1.0 - eps)
    if minval is not None and maxval is not None:
      vector = asserts.assert_all_in_range(
          vector, minval, maxval, open_bounds=open_bounds)
    return vector


def safe_signed_div(a, b, eps=None, name='safe_signed_div'):
  """Calculates a/b safely.

  If the tf-graphics debug flag is set to `True`, this function adds assertions
  to the graph that check whether `abs(b + eps)` is greather than zero, and the
  division has no `NaN` or `Inf` values.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    a: A `float` or a tensor of shape `[A1, ..., An]`, which is the nominator.
    b: A `float` or a tensor of shape `[A1, ..., An]`, which is the denominator
      with non-negative values.
    eps: A small `float`, to be added to the denominator. If left `None`, its
      value is automatically selected using `b.dtype`.
    name: A name for this op. Defaults to 'safe_signed_div'.

  Raises:
     InvalidArgumentError: If tf-graphics debug flag is set and the division
       causes `NaN` or `Inf` values.

  Returns:
     A tensor of shape `[A1, ..., An]` containing the results of division.
  """
  with tf.name_scope(name):
    a = tf.convert_to_tensor(value=a)
    b = tf.convert_to_tensor(value=b)
    if eps is None:
      eps = asserts.select_eps_for_division(b.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=b.dtype)

    return asserts.assert_no_infs_or_nans(a / (b + nonzero_sign(b) * eps))


def safe_sinpx_div_sinx(theta, factor, eps=None, name='safe_sinpx_div_sinx'):
  """Calculates sin(factor * theta)/sin(theta) safely.

  The term `sin(factor * theta)/sin(theta)` appears when calculating spherical
  interpolation weights, and it has periodic edge cases causing both zero / zero
  and division by zero problems. This function adds signed eps to the angles in
  both the nominator and the denominator to ensure safety, and returns the
  correct values estimated by l'Hopital rule in the case of zero / zero.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    theta: A tensor of shape `[A1, ..., An]` representing angles in radians.
    factor: A `float` or a tensor of shape `[A1, ..., An]`.
    eps: A `float`, used to perturb the angle. If left as `None`, its value is
      automatically determined from the `dtype` of `theta`.
    name: A name for this op. Defaults to 'safe_sinpx_div_sinx'.

  Raises:
    InvalidArgumentError: If tf-graphics debug flag is set and the division
      returns `NaN` or `Inf` values.

  Returns:
    A tensor of shape `[A1, ..., An]` containing the resulting values.
  """
  with tf.name_scope(name):
    theta = tf.convert_to_tensor(value=theta)
    factor = tf.convert_to_tensor(value=factor, dtype=theta.dtype)
    if eps is None:
      eps = asserts.select_eps_for_division(theta.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=theta.dtype)

    # eps will be multiplied with factor next, which can make it zero.
    # Therefore we multiply eps with min(1/factor, 1e10), which can handle
    # factors as small as 1e-10 correctly, while preventing a division by zero.
    eps *= tf.clip_by_value(1.0 / factor, 1.0, 1e10)
    sign = nonzero_sign(0.5 * np.pi - theta % np.pi)
    theta += sign * eps
    div = tf.sin(factor * theta) / tf.sin(theta)
    return asserts.assert_no_infs_or_nans(div)


def safe_unsigned_div(a, b, eps=None, name='safe_unsigned_div'):
  """Calculates a/b with b >= 0 safely.

  If the tfg debug flag TFG_ADD_ASSERTS_TO_GRAPH defined in tfg_flags.py
  is set to True, this function adds assertions to the graph that check whether
  b + eps is greather than zero, and the division has no NaN or Inf values.

  Args:
    a: A `float` or a tensor of shape `[A1, ..., An]`, which is the nominator.
    b: A `float` or a tensor of shape `[A1, ..., An]`, which is the denominator.
    eps: A small `float`, to be added to the denominator. If left as `None`, its
      value is automatically selected using `b.dtype`.
    name: A name for this op. Defaults to 'safe_unsigned_div'.

  Raises:
     InvalidArgumentError: If tf-graphics debug flag is set and the division
       causes `NaN` or `Inf` values.

  Returns:
     A tensor of shape `[A1, ..., An]` containing the results of division.
  """
  with tf.name_scope(name):
    a = tf.convert_to_tensor(value=a)
    b = tf.convert_to_tensor(value=b)
    if eps is None:
      eps = asserts.select_eps_for_division(b.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=b.dtype)

    return asserts.assert_no_infs_or_nans(a / (b + eps))


# The util functions or classes are not exported.
__all__ = []
