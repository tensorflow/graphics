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
"""Assert functions to be used by various modules.

This module contains asserts that are intended to be used in TensorFlow
Graphics. These asserts will be activated only if the debug flag
TFG_ADD_ASSERTS_TO_GRAPH is set to True.
"""

from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_graphics.util import tfg_flags

FLAGS = flags.FLAGS


def assert_no_infs_or_nans(tensor, name='assert_no_infs_or_nans'):
  """Checks a tensor for NaN and Inf values.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    tensor: A tensor of shape `[A1, ..., An]` containing the values we want to
      check.
    name: A name for this op. Defaults to 'assert_no_infs_or_nans'.

  Raises:
    tf.errors.InvalidArgumentError: If any entry of the input is NaN or Inf.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return tensor

  with tf.name_scope(name):
    tensor = tf.convert_to_tensor(value=tensor)

    assert_ops = (tf.debugging.check_numerics(
        tensor, message='Inf or NaN detected.'),)
    with tf.control_dependencies(assert_ops):
      return tf.identity(tensor)


def assert_all_above(vector, minval, open_bound=False, name='assert_all_above'):
  """Checks whether all values of vector are above minval.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector: A tensor of shape `[A1, ..., An]` containing the values we want to
      check.
    minval: A scalar or a tensor of shape `[A1, ..., An]` representing the
      desired lower bound for the values in `vector`.
    open_bound: A `bool` indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_above'.

  Raises:
    tf.errors.InvalidArgumentError: If any entry of the input is below `minval`.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)

    if open_bound:
      assert_ops = (tf.debugging.assert_greater(vector, minval),)
    else:
      assert_ops = (tf.debugging.assert_greater_equal(vector, minval),)
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_all_below(vector, maxval, open_bound=False, name='assert_all_below'):
  """Checks whether all values of vector are below maxval.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector: A tensor of shape `[A1, ..., An]` containing the values we want to
      check.
    maxval: A scalar or a tensor of shape `[A1, ..., An]` representing the
      desired upper bound for the values in `vector`.
    open_bound: A boolean indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_below'.

  Raises:
    tf.errors.InvalidArgumentError: If any entry of the input exceeds `maxval`.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)

    if open_bound:
      assert_ops = (tf.debugging.assert_less(vector, maxval),)
    else:
      assert_ops = (tf.debugging.assert_less_equal(vector, maxval),)
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_all_in_range(vector,
                        minval,
                        maxval,
                        open_bounds=False,
                        name='assert_all_in_range'):
  """Checks whether all values of vector are between minval and maxval.

  This function checks if all the values in the given vector are in an interval
  `[minval, maxval]` if `open_bounds` is `False`, or in `]minval, maxval[` if it
  is set to `True`.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector: A tensor of shape `[A1, ..., An]` containing the values we want to
      check.
    minval: A `float` or a tensor of shape `[A1, ..., An]` representing the
      desired lower bound for the values in `vector`.
    maxval: A `float` or a tensor of shape `[A1, ..., An]` representing the
      desired upper bound for the values in `vector`.
    open_bounds: A `bool` indicating whether the range is open or closed.
    name: A name for this op. Defaults to 'assert_all_in_range'.

  Raises:
    tf.errors.InvalidArgumentError: If `vector` is not in the expected range.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
    maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)

    if open_bounds:
      assert_ops = (tf.debugging.assert_less(vector, maxval),
                    tf.debugging.assert_greater(vector, minval))
    else:
      assert_ops = (tf.debugging.assert_less_equal(vector, maxval),
                    tf.debugging.assert_greater_equal(vector, minval))
    with tf.control_dependencies(assert_ops):
      return tf.identity(vector)


def assert_nonzero_norm(vector, eps=None, name='assert_nonzero_norm'):
  """Checks whether vector/quaternion has non-zero norm in its last dimension.

  This function checks whether all the norms of the vectors are greater than
  eps, such that normalizing them will not generate NaN values. Normalization is
  assumed to be done in the last dimension of vector. If eps is left as `None`,
  the function will determine the most suitable value depending on the `dtype`
  of the `vector`.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector: A tensor of shape `[A1, ..., An, V]`, where the last dimension
      contains a V dimensional vector.
    eps: A `float` describing the tolerance used to determine if the norm is
      equal to zero.
    name: A name for this op. Defaults to 'assert_nonzero_norm'.

  Raises:
    InvalidArgumentError: If `vector` has zero norm.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    if eps is None:
      eps = select_eps_for_division(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

    norm = tf.norm(tensor=vector, axis=-1)
    with tf.control_dependencies([tf.debugging.assert_greater(norm, eps)]):
      return tf.identity(vector)


def assert_normalized(vector,
                      order='euclidean',
                      axis=-1,
                      eps=None,
                      name='assert_normalized'):
  """Checks whether vector/quaternion is normalized in its last dimension.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vector: A tensor of shape `[A1, ..., M, ..., An]`, where the axis of M
      contains the vectors.
    order: Order of the norm passed to tf.norm.
    axis: The axis containing the vectors.
    eps: A `float` describing the tolerance used to determine if the norm is
      equal to `1.0`.
    name: A name for this op. Defaults to 'assert_normalized'.

  Raises:
    InvalidArgumentError: If the norm of `vector` is not `1.0`.

  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return vector

  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    if eps is None:
      eps = select_eps_for_division(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

    norm = tf.norm(tensor=vector, ord=order, axis=axis)
    one = tf.constant(1.0, dtype=norm.dtype)
    with tf.control_dependencies(
        [tf.debugging.assert_near(norm, one, atol=eps)]):
      return tf.identity(vector)


def assert_at_least_k_non_zero_entries(tensor,
                                       k=1,
                                       name='assert_at_least_k_non_zero_entries'
                                      ):
  """Checks if `tensor` has at least k non-zero entries in the last dimension.

  Given a tensor with `M` dimensions in its last axis, this function checks
  whether at least `k` out of `M` dimensions are non-zero.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    tensor: A tensor of shape `[A1, ..., An, M]`.
    k: An integer, corresponding to the minimum number of non-zero entries.
    name: A name for this op. Defaults to 'assert_at_least_k_non_zero_entries'.

  Raises:
    InvalidArgumentError: If `tensor` has less than `k` non-zero entries in its
      last axis.

  Returns:
    The input tensor, with dependence on the assertion operator in the graph.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return tensor

  with tf.name_scope(name):
    tensor = tf.convert_to_tensor(value=tensor)

    indicator = tf.cast(tf.math.greater(tensor, 0.0), dtype=tensor.dtype)
    indicator_sum = tf.reduce_sum(input_tensor=indicator, axis=-1)
    assert_op = tf.debugging.assert_greater_equal(
        indicator_sum, tf.cast(k, dtype=tensor.dtype))
    with tf.control_dependencies([assert_op]):
      return tf.identity(tensor)


def assert_binary(tensor, name='assert_binary'):
  """Asserts that all the values in the tensor are zeros or ones.

  Args:
    tensor: A tensor of shape `[A1, ..., An]` containing the values we want to
      check.
    name: A name for this op. Defaults to 'assert_binary'.

  Returns:
    The input tensor, with dependence on the assertion operator in the graph.

  Raises:
    tf.errors.InvalidArgumentError: If any of the values in the tensor is not
    zero or one.
  """
  if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    return tensor

  with tf.name_scope(name):
    tensor = tf.convert_to_tensor(value=tensor)
    condition = tf.reduce_all(
        input_tensor=tf.logical_or(tf.equal(tensor, 0), tf.equal(tensor, 1)))

    with tf.control_dependencies(
        [tf.debugging.Assert(condition, data=[tensor])]):
      return tf.identity(tensor)


def select_eps_for_addition(dtype):
  """Returns 2 * machine epsilon based on `dtype`.

  This function picks an epsilon slightly greater than the machine epsilon,
  which is the upper bound on relative error. This value ensures that
  `1.0 + eps != 1.0`.

  Args:
    dtype: The `tf.DType` of the tensor to which eps will be added.

  Raises:
    ValueError: If `dtype` is not a floating type.

  Returns:
    A `float` to be used to make operations safe.
  """
  return 2.0 * np.finfo(dtype.as_numpy_dtype).eps


def select_eps_for_division(dtype):
  """Selects default values for epsilon to make divisions safe based on dtype.

  This function returns an epsilon slightly greater than the smallest positive
  floating number that is representable for the given dtype. This is mainly used
  to prevent division by zero, which produces Inf values. However, if the
  nominator is orders of magnitude greater than `1.0`, eps should also be
  increased accordingly. Only floating types are supported.

  Args:
    dtype: The `tf.DType` of the tensor to which eps will be added.

  Raises:
    ValueError: If `dtype` is not a floating type.

  Returns:
    A `float` to be used to make operations safe.
  """
  return 10.0 * np.finfo(dtype.as_numpy_dtype).tiny


# The util functions or classes are not exported.
__all__ = []
