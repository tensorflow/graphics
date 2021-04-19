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
"""Shape utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf


def _broadcast_shape_helper(shape_x, shape_y):
  """Helper function for is_broadcast_compatible and broadcast_shape.

  Args:
    shape_x: A `TensorShape`.
    shape_y: A `TensorShape`.

  Returns:
    Returns None if the shapes are not broadcast compatible, or a list
    containing the broadcasted dimensions otherwise.
  """
  # To compute the broadcasted dimensions, we zip together shape_x and shape_y,
  # and pad with 1 to make them the same length.
  broadcasted_dims = reversed(
      list(
          six.moves.zip_longest(
              reversed(shape_x.dims),
              reversed(shape_y.dims),
              fillvalue=tf.compat.v1.Dimension(1))))
  # Next we combine the dimensions according to the numpy broadcasting rules.
  # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      # One or both dimensions is unknown. If either dimension is greater than
      # 1, we assume that the program is correct, and the other dimension will
      # be broadcast to match it.
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      # We will broadcast dim_x to dim_y.
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      # We will broadcast dim_y to dim_x.
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      # The dimensions are compatible, so output is the same size in that
      # dimension.
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      return None
  return return_dims


def is_broadcast_compatible(shape_x, shape_y):
  """Returns True if `shape_x` and `shape_y` are broadcast compatible.

  Args:
    shape_x: A `TensorShape`.
    shape_y: A `TensorShape`.

  Returns:
    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted
    to. False otherwise.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return False
  return _broadcast_shape_helper(shape_x, shape_y) is not None


def get_broadcasted_shape(shape_x, shape_y):
  """Returns the common shape for broadcast compatible shapes.

  Args:
    shape_x: A `TensorShape`.
    shape_y: A `TensorShape`.

  Returns:
    Returns None if the shapes are not broadcast compatible, or a list
    containing the broadcasted dimensions otherwise.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return None
  return _broadcast_shape_helper(shape_x, shape_y)


def _check_type(variable, variable_name, expected_type):
  """Helper function for checking that inputs are of expected types."""
  if isinstance(expected_type, (list, tuple)):
    expected_type_name = 'list or tuple'
  else:
    expected_type_name = expected_type.__name__
  if not isinstance(variable, expected_type):
    raise ValueError('{} must be of type {}, but it is {}'.format(
        variable_name, expected_type_name,
        type(variable).__name__))


def _fix_axis_dim_pairs(pairs, name):
  """Helper function to make `pairs` a list if needed."""
  if isinstance(pairs[0], int):
    pairs = [pairs]
  for pair in pairs:
    if len(pair) != 2:
      raise ValueError(
          '{} must consist of axis-value pairs, but found {}'.format(
              name, pair))
  return pairs


def _get_dim(tensor, axis):
  """Returns dimensionality of a tensor for a given axis."""
  return tf.compat.dimension_value(tensor.shape[axis])


def check_static(tensor,
                 has_rank=None,
                 has_rank_greater_than=None,
                 has_rank_less_than=None,
                 has_dim_equals=None,
                 has_dim_greater_than=None,
                 has_dim_less_than=None,
                 tensor_name='tensor'):
  """Checks static shapes for rank and dimension constraints.

  This function can be used to check a tensor's shape for multiple rank and
  dimension constraints at the same time.

  Args:
    tensor: Any tensor with a static shape.
    has_rank: An int or `None`. If not `None`, the function checks if the rank
      of the `tensor` equals to `has_rank`.
    has_rank_greater_than: An int or `None`. If not `None`, the function checks
      if the rank of the `tensor` is greater than `has_rank_greater_than`.
    has_rank_less_than: An int or `None`. If not `None`, the function checks if
      the rank of the `tensor` is less than `has_rank_less_than`.
    has_dim_equals: Either a tuple or list containing a single pair of `int`s,
      or a list or tuple containing multiple such pairs. Each pair is in the
      form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] == dim`.
    has_dim_greater_than: Either a tuple or list containing a single pair of
      `int`s, or a list or tuple containing multiple such pairs. Each pair is in
      the form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] > dim`.
    has_dim_less_than: Either a tuple or list containing a single pair of
      `int`s, or a list or tuple containing multiple such pairs. Each pair is in
      the form (`axis`, `dim`), which means the function should check if
      `tensor.shape[axis] < dim`.
    tensor_name: A name for `tensor` to be used in the error message if one is
      thrown.

  Raises:
    ValueError: If any input is not of the expected types, or if one of the
      checks described above fails.
  """
  rank = tensor.shape.ndims

  def _raise_value_error_for_rank(variable, error_msg):
    raise ValueError(
        '{} must have a rank {} {}, but it has rank {} and shape {}'.format(
            tensor_name, error_msg, variable, rank, tensor.shape.as_list()))

  def _raise_value_error_for_dim(tensor_name, error_msg, axis, value):
    raise ValueError(
        '{} must have {} {} dimensions in axis {}, but it has shape {}'.format(
            tensor_name, error_msg, value, axis, tensor.shape.as_list()))

  if has_rank is not None:
    _check_type(has_rank, 'has_rank', int)
    if rank != has_rank:
      _raise_value_error_for_rank(has_rank, 'of')
  if has_rank_greater_than is not None:
    _check_type(has_rank_greater_than, 'has_rank_greater_than', int)
    if rank <= has_rank_greater_than:
      _raise_value_error_for_rank(has_rank_greater_than, 'greater than')
  if has_rank_less_than is not None:
    _check_type(has_rank_less_than, 'has_rank_less_than', int)
    if rank >= has_rank_less_than:
      _raise_value_error_for_rank(has_rank_less_than, 'less than')
  if has_dim_equals is not None:
    _check_type(has_dim_equals, 'has_dim_equals', (list, tuple))
    has_dim_equals = _fix_axis_dim_pairs(has_dim_equals, 'has_dim_equals')
    for axis, value in has_dim_equals:
      if _get_dim(tensor, axis) != value:
        _raise_value_error_for_dim(tensor_name, 'exactly', axis, value)
  if has_dim_greater_than is not None:
    _check_type(has_dim_greater_than, 'has_dim_greater_than', (list, tuple))
    has_dim_greater_than = _fix_axis_dim_pairs(has_dim_greater_than,
                                               'has_dim_greater_than')
    for axis, value in has_dim_greater_than:
      if not _get_dim(tensor, axis) > value:
        _raise_value_error_for_dim(tensor_name, 'greater than', axis, value)
  if has_dim_less_than is not None:
    _check_type(has_dim_less_than, 'has_dim_less_than', (list, tuple))
    has_dim_less_than = _fix_axis_dim_pairs(has_dim_less_than,
                                            'has_dim_less_than')
    for axis, value in has_dim_less_than:
      if not _get_dim(tensor, axis) < value:
        _raise_value_error_for_dim(tensor_name, 'less than', axis, value)


def _check_tensors(tensors, tensors_name):
  """Helper function to check the type and length of tensors."""
  _check_type(tensors, tensors_name, (list, tuple))
  if len(tensors) < 2:
    raise ValueError('At least 2 tensors are required.')


def _check_tensor_axis_lists(tensors, tensors_name, axes, axes_name):
  """Helper function to check that lengths of `tensors` and `axes` match."""
  _check_type(axes, axes_name, (list, tuple))
  if len(tensors) != len(axes):
    raise ValueError(
        '{} and {} must have the same length, but are {} and {}.'.format(
            tensors_name, axes_name, len(tensors), len(axes)))


def _fix_axes(tensors, axes, allow_negative):
  """Makes all axes positive and checks for out of bound errors."""
  axes = [
      axis + tensor.shape.ndims if axis < 0 else axis
      for tensor, axis in zip(tensors, axes)
  ]
  if not all(
      ((allow_negative or
        (not allow_negative and axis >= 0)) and axis < tensor.shape.ndims)
      for tensor, axis in zip(tensors, axes)):
    rank_axis_pairs = list(
        zip([tensor.shape.ndims for tensor in tensors], axes))
    raise ValueError(
        'Some axes are out of bounds. Given rank-axes pairs: {}'.format(
            [pair for pair in rank_axis_pairs]))
  return axes


def _give_default_names(list_of_objects, name):
  """Helper function to give default names to objects for error messages."""
  return [name + '_' + str(index) for index in range(len(list_of_objects))]


def _all_are_equal(list_of_objects):
  """Helper function to check if all the items in a list are the same."""
  if not list_of_objects:
    return True
  if isinstance(list_of_objects[0], list):
    list_of_objects = [tuple(obj) for obj in list_of_objects]
  return len(set(list_of_objects)) == 1


def _raise_error(tensor_names, batch_shapes):
  formatted_list = [(name, batch_shape)
                    for name, batch_shape in zip(tensor_names, batch_shapes)]
  raise ValueError(
      'Not all batch dimensions are identical: {}'.format(formatted_list))


def compare_batch_dimensions(tensors,
                             last_axes,
                             broadcast_compatible,
                             initial_axes=0,
                             tensor_names=None):
  """Compares batch dimensions for tensors with static shapes.

  Args:
    tensors: A list or tuple of tensors with static shapes to compare.
    last_axes: An `int` or a list or tuple of `int`s with the same length as
      `tensors`. If an `int`, it is assumed to be the same for all the tensors.
      Each entry should correspond to the last axis of the batch (with zero
      based indices). For instance, if there is only a single batch dimension,
      last axis should be `0`.
    broadcast_compatible: A 'bool', whether the batch shapes can be broadcast
      compatible in the numpy sense.
    initial_axes: An `int` or a list or tuple of `int`s with the same length as
      `tensors`. If an `int`, it is assumed to be the same for all the tensors.
      Each entry should correspond to the first axis of the batch (with zero
      based indices). Default value is `0`.
    tensor_names: Names of `tensors` to be used in the error message if one is
      thrown. If left as `None`, `tensor_i` is used.

  Raises:
    ValueError: If inputs have unexpected types, or if given axes are out of
      bounds, or if the check fails.
  """
  _check_tensors(tensors, 'tensors')
  if isinstance(initial_axes, int):
    initial_axes = [initial_axes] * len(tensors)
  if isinstance(last_axes, int):
    last_axes = [last_axes] * len(tensors)
  _check_tensor_axis_lists(tensors, 'tensors', initial_axes, 'initial_axes')
  _check_tensor_axis_lists(tensors, 'tensors', last_axes, 'last_axes')
  initial_axes = _fix_axes(tensors, initial_axes, allow_negative=True)
  last_axes = _fix_axes(tensors, last_axes, allow_negative=True)
  batch_shapes = [
      tensor.shape[init:last + 1]
      for tensor, init, last in zip(tensors, initial_axes, last_axes)
  ]
  if tensor_names is None:
    tensor_names = _give_default_names(tensors, 'tensor')
  if not broadcast_compatible:
    batch_ndims = [batch_shape.ndims for batch_shape in batch_shapes]
    batch_shapes = [batch_shape.as_list() for batch_shape in batch_shapes]
    if not _all_are_equal(batch_ndims):
      # If not all batch shapes have the same length, they cannot be identical.
      _raise_error(tensor_names, batch_shapes)
    for dims in zip(*batch_shapes):
      if _all_are_equal(dims):
        # Continue if all dimensions are None or have the same value.
        continue
      if None not in dims:
        # If all dimensions are known at this point, they are not identical.
        _raise_error(tensor_names, batch_shapes)
      # At this point dims must consist of both None's and int's.
      if len(set(dims)) != 2:
        # set(dims) should return (None, some_int).
        # Otherwise shapes are not identical.
        _raise_error(tensor_names, batch_shapes)
  else:
    if not all(
        is_broadcast_compatible(shape1, shape2)
        for shape1, shape2 in itertools.combinations(batch_shapes, 2)):
      raise ValueError(
          'Not all batch dimensions are broadcast-compatible: {}'.format([
              (name, batch_shape.as_list())
              for name, batch_shape in zip(tensor_names, batch_shapes)
          ]))


def compare_dimensions(tensors, axes, tensor_names=None):
  """Compares dimensions of tensors with static or dynamic shapes.

  Args:
    tensors: A list or tuple of tensors to compare.
    axes: An `int` or a list or tuple of `int`s with the same length as
      `tensors`. If an `int`, it is assumed to be the same for all the tensors.
      Each entry should correspond to the axis of the tensor being compared.
    tensor_names: Names of `tensors` to be used in the error message if one is
      thrown. If left as `None`, their `Tensor.name` fields are used instead.

  Raises:
    ValueError: If inputs have unexpected types, or if given axes are out of
      bounds, or if the check fails.
  """
  _check_tensors(tensors, 'tensors')
  if isinstance(axes, int):
    axes = [axes] * len(tensors)
  _check_tensor_axis_lists(tensors, 'tensors', axes, 'axes')
  axes = _fix_axes(tensors, axes, allow_negative=False)
  if tensor_names is None:
    tensor_names = _give_default_names(tensors, 'tensor')
  dimensions = [_get_dim(tensor, axis) for tensor, axis in zip(tensors, axes)]
  if not _all_are_equal(dimensions):
    raise ValueError('Tensors {} must have the same number of dimensions in '
                     'axes {}, but they are {}.'.format(
                         list(tensor_names), list(axes), list(dimensions)))


def is_static(tensor_shape):
  """Checks if the given tensor shape is static."""
  if isinstance(tensor_shape, (list, tuple)):
    return None not in tensor_shape
  else:
    return None not in tensor_shape.as_list()


def add_batch_dimensions(tensor, tensor_name, batch_shape, last_axis=None):
  """Broadcasts tensor to match batch dimensions.

  It will either broadcast to all provided batch dimensions, therefore
  increasing tensor shape by len(batch_shape) dimensions or will do nothing if
  batch dimensions already present and equal to expected batch dimensions.

  Args:
    tensor: A tensor to broadcast of a shape [A1, ..., An, B1, ..., Bn]. Where
      [A1, ..., An] is batch dimensions (it is allowed to have no batch
      dimensions), and [B1, ..., Bn] are other tensor dimensions. If [A1, ...,
      An] are present but different from values in `batch_shape` the error will
      be thrown.
    tensor_name: Name of `tensor` to be used in the error message if one is
    batch_shape: list of `int` representing desired batch dimensions.
    last_axis: An `int` corresponding to the last axis of the batch (with zero
      based indices). For instance, if there is only a single batch dimension,
      last axis should be `0`. If there is no batch dimensions it must be set to
      `None`. thrown.

  Returns:
    Tensor of a shape `batch_shape` + [B1, ..., Bn] or unmodified tensor if
    `batch_shape` = [A1, ..., An].
  Raises:
    ValueError if tensor already has batch dimensions different from desired
      one.
  """
  if last_axis is not None:
    last_axis = _fix_axes([tensor], [last_axis], allow_negative=True)[0]
    tensor_batch_shape = tensor.shape.as_list()[:last_axis + 1]
    if np.array_equal(tensor_batch_shape, batch_shape):
      return tensor
    elif tensor_batch_shape:
      raise ValueError(
          'Tensor {} has batch dimensions different from target '
          'one. Found {}, but expected no batch dimensions or {}'.format(
              tensor_name, tensor.shape[:last_axis + 1], batch_shape))

  return tf.broadcast_to(tensor, batch_shape + list(tensor.shape))


# The util functions or classes are not exported.
__all__ = []
