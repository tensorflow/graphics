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
"""This module implements various sparse data utilities for graphs and meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, List, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def _is_dynamic_shape(tensors: Union[List[type_alias.TensorLike],
                                     Tuple[Any, tf.sparse.SparseTensor]]):
  """Helper function to test if any tensor in a list has a dynamic shape.

  Args:
    tensors: A list or tuple of tensors with shapes to test.

  Returns:
    True if any tensor in the list has a dynamic shape, False otherwise.
  """
  if not isinstance(tensors, (list, tuple)):
    raise ValueError("'tensors' must be list of tuple.")
  return not all([shape.is_static(tensor.shape) for tensor in tensors])


def check_valid_graph_convolution_input(data: type_alias.TensorLike,
                                        neighbors: tf.sparse.SparseTensor,
                                        sizes: type_alias.TensorLike):
  """Checks that the inputs are valid for graph convolution ops.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V1, V2]`.
    neighbors: A SparseTensor with the same type as `data` and with shape `[A1,
      ..., An, V1, V1]`.
    sizes: An `int` tensor of shape `[A1, ..., An]`. Optional, can be `None`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  if not data.dtype.is_floating:
    raise TypeError("'data' must have a float type.")
  if neighbors.dtype != data.dtype:
    raise TypeError("'neighbors' and 'data' must have the same type.")
  if sizes is not None and not sizes.dtype.is_integer:
    raise TypeError("'sizes' must have an integer type.")
  if not isinstance(neighbors, tf.sparse.SparseTensor):
    raise ValueError("'neighbors' must be a SparseTensor.")

  data_ndims = data.shape.ndims
  shape.check_static(tensor=data, tensor_name="data", has_rank_greater_than=1)
  shape.check_static(
      tensor=neighbors, tensor_name="neighbors", has_rank=data_ndims)
  if not _is_dynamic_shape(tensors=(data, neighbors)):
    shape.compare_dimensions(
        tensors=(data, neighbors, neighbors),
        tensor_names=("data", "neighbors", "neighbors"),
        axes=(-2, -2, -1))
  if sizes is None:
    shape.compare_batch_dimensions(
        tensors=(data, neighbors),
        tensor_names=("data", "neighbors"),
        last_axes=-3,
        broadcast_compatible=False)
  else:
    shape.check_static(
        tensor=sizes, tensor_name="sizes", has_rank=data_ndims - 2)
    shape.compare_batch_dimensions(
        tensors=(data, neighbors, sizes),
        tensor_names=("data", "neighbors", "sizes"),
        last_axes=(-3, -3, -1),
        broadcast_compatible=False)


def check_valid_graph_pooling_input(data: type_alias.TensorLike,
                                    pool_map: tf.sparse.SparseTensor,
                                    sizes: type_alias.TensorLike):
  """Checks that the inputs are valid for graph pooling.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V1, C]`.
    pool_map: A SparseTensor with the same type as `data` and with shape `[A1,
      ..., An, V2, V1]`.
    sizes: An `int` tensor of shape `[A1, ..., An, 2]`. Can be `None`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  if not data.dtype.is_floating:
    raise TypeError("'data' must have a float type.")
  if pool_map.dtype != data.dtype:
    raise TypeError("'pool_map' and 'data' must have the same type.")
  if sizes is not None and not sizes.dtype.is_integer:
    raise TypeError("'sizes' must have an integer type.")
  if not isinstance(pool_map, tf.sparse.SparseTensor):
    raise ValueError("'pool_map' must be a SparseTensor.")

  data_ndims = data.shape.ndims
  shape.check_static(tensor=data, tensor_name="data", has_rank_greater_than=1)
  shape.check_static(
      tensor=pool_map, tensor_name="pool_map", has_rank=data_ndims)
  if not _is_dynamic_shape(tensors=(data, pool_map)):
    shape.compare_dimensions(
        tensors=(data, pool_map),
        tensor_names=("data", "pool_map"),
        axes=(-2, -1))
  if sizes is None:
    shape.compare_batch_dimensions(
        tensors=(data, pool_map),
        tensor_names=("data", "pool_map"),
        last_axes=-3,
        broadcast_compatible=False)
  else:
    shape.check_static(
        tensor=sizes, tensor_name="sizes", has_rank=data_ndims - 1)
    shape.compare_batch_dimensions(
        tensors=(data, pool_map, sizes),
        tensor_names=("data", "pool_map", "sizes"),
        last_axes=(-3, -3, -2),
        broadcast_compatible=False)


def check_valid_graph_unpooling_input(data: type_alias.TensorLike,
                                      pool_map: tf.sparse.SparseTensor,
                                      sizes: type_alias.TensorLike):
  """Checks that the inputs are valid for graph unpooling.

  Note:
    In the following, A1 to A3 are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
    pool_map: A `SparseTensor` with the same type as `data` and with shape `[A1,
      ..., A3, V1, V2]`.
    sizes: An `int` tensor of shape `[A1, ..., A3, 2]`. Can be `None`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  if not data.dtype.is_floating:
    raise TypeError("'data' must have a float type.")
  if pool_map.dtype != data.dtype:
    raise TypeError("'pool_map' and 'data' must have the same type.")
  if sizes is not None and not sizes.dtype.is_integer:
    raise TypeError("'sizes' must have an integer type.")
  if not isinstance(pool_map, tf.sparse.SparseTensor):
    raise ValueError("'pool_map' must be a SparseTensor.")

  data_ndims = data.shape.ndims
  shape.check_static(tensor=data, tensor_name="data", has_rank_greater_than=1)
  shape.check_static(tensor=data, tensor_name="data", has_rank_less_than=6)
  shape.check_static(
      tensor=pool_map, tensor_name="pool_map", has_rank=data_ndims)
  if not _is_dynamic_shape(tensors=(data, pool_map)):
    shape.compare_dimensions(
        tensors=(data, pool_map),
        tensor_names=("data", "pool_map"),
        axes=(-2, -2))
  if sizes is None:
    shape.compare_batch_dimensions(
        tensors=(data, pool_map),
        tensor_names=("data", "pool_map"),
        last_axes=-3,
        broadcast_compatible=False)
  else:
    shape.check_static(
        tensor=sizes, tensor_name="sizes", has_rank=data_ndims - 1)
    shape.compare_batch_dimensions(
        tensors=(data, pool_map, sizes),
        tensor_names=("data", "pool_map", "sizes"),
        last_axes=(-3, -3, -2),
        broadcast_compatible=False)


def flatten_batch_to_2d(data: type_alias.TensorLike,
                        sizes: type_alias.TensorLike = None,
                        name: str = "utils_flatten_batch_to_2d"):
  """Reshapes a batch of 2d Tensors by flattening across the batch dimensions.

  Note:
    In the following, A1 to An are optional batch dimensions.

  A tensor with shape `[A1, ..., An, D1, D2]` will be reshaped to one
  with shape `[A1*...*An*D1, D2]`. This function also returns an inverse
  function that returns any tensor with shape `[A1*...*An*D1, D3]` to one
  with shape `[A1, ..., An, D1, D3]`.

  Padded inputs in dimension D1 are allowed. `sizes` determines the first
  elements from D1 to select from each batch dimension.

  Examples:
    ```python
    data = [[[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]]]
    sizes = None
    output = flatten_batch_to_2d(data, size)
    print(output)
    >>> [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]]

    data = [[[1., 2.], [0., 0.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [0., 0.]]]
    sizes = [1, 2, 1]
    output = flatten_batch_to_2d(data, size)
    print(output)
    >>> [[1., 2.], [5., 6.], [7., 8.], [9., 10.]]
    ```

  Args:
    data: A tensor with shape `[A1, ..., An, D1, D2]`.
    sizes: An `int` tensor with shape `[A1, ..., An]`. Can be `None`. `sizes[i]
      <= D1`.
    name: A name for this op. Defaults to 'utils_flatten_batch_to_2d'.

  Returns:
    A tensor with shape `[A1*...*An*D1, D2]` if `sizes == None`, otherwise a
      tensor  with shape `[sum(sizes), D2]`.
    A function that reshapes a tensor with shape `[A1*...*An*D1, D3]` to a
      tensor with shape `[A1, ..., An, D1, D3]` if `sizes == None`, otherwise
      it reshapes a tensor with shape `[sum(sizes), D3]` to one with shape
      `[A1, ..., An, ..., D1, D3]`.

  Raises:
    ValueError: if the input tensor dimensions are invalid.
  """
  with tf.name_scope(name):
    data = tf.convert_to_tensor(value=data)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)

    if sizes is not None and not sizes.dtype.is_integer:
      raise TypeError("'sizes' must have an integer type.")
    shape.check_static(tensor=data, tensor_name="data", has_rank_greater_than=2)
    if sizes is not None:
      shape.check_static(
          tensor=sizes, tensor_name="sizes", has_rank=data.shape.ndims - 2)
      shape.compare_batch_dimensions(
          tensors=(data, sizes),
          tensor_names=("data", "sizes"),
          last_axes=(-3, -1),
          broadcast_compatible=False)

    data_shape = tf.shape(input=data)
    if sizes is None:
      flat = tf.reshape(data, shape=(-1, data_shape[-1]))

      def unflatten(flat, name="utils_unflatten"):
        """Invert flatten_batch_to_2d."""
        with tf.name_scope(name):
          flat = tf.convert_to_tensor(value=flat)
          output_shape = tf.concat((data_shape[:-1], tf.shape(input=flat)[-1:]),
                                   axis=0)
          return tf.reshape(flat, output_shape)
    else:
      # Create a mask for the desired rows in `data` to select for flattening:
      # `mask` has shape `[A1, ..., An, D1]` and
      # `mask[a1, ..., an, :] = [True, ..., True, False, ..., False]` where
      # the number of True elements is `sizes[a1, ..., an]`.
      mask = tf.sequence_mask(sizes, data_shape[-2])
      mask_indices = tf.cast(tf.where(mask), tf.int32)
      flat = tf.gather_nd(params=data, indices=mask_indices)

      def unflatten(flat, name="utils_unflatten"):
        """Invert flatten_batch_to_2d."""
        with tf.name_scope(name):
          flat = tf.convert_to_tensor(value=flat)
          output_shape = tf.concat((data_shape[:-1], tf.shape(input=flat)[-1:]),
                                   axis=0)
          return tf.scatter_nd(
              indices=mask_indices, updates=flat, shape=output_shape)

    return flat, unflatten


def unflatten_2d_to_batch(data: type_alias.TensorLike,
                          sizes: type_alias.TensorLike,
                          max_rows: Optional[int] = None,
                          name: str = "utils_unflatten_2d_to_batch"):
  r"""Reshapes a 2d Tensor into a batch of 2d Tensors.

  The `data` tensor with shape `[D1, D2]` will be mapped to a tensor with shape
  `[A1, ..., An, max_rows, D2]` where `max_rows` defaults to `max(sizes)`.
  `sizes` determines the segment of rows in the input that get mapped to a
  particular batch dimension (`sum(sizes) == D1`).

  Examples:

    ```python
    data = [[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.],
            [9., 10.],
            [11., 12.]]
    sizes = [2, 3, 1]

    output = unflatten_2d_to_batch(data, sizes, max_rows=None)
    print(output.shape)
    >>> [3, 3, 2]
    print(output)
    >>> [[[1., 2.],
          [3., 4.],
          [0., 0.]],
         [[5., 6.],
          [7., 8.],
          [9., 10.]],
         [[11., 12.],
          [0., 0.],
          [0., 0.]]]

    output = unflatten_2d_to_batch(data, sizes, max_rows=4)
    print(output.shape)
    >>> [3, 4, 2]
    print(output)
    >>> [[[1., 2.],
          [3., 4.],
          [0., 0.],
          [0., 0.]],
         [[5., 6.],
          [7., 8.],
          [9., 10.],
          [0., 0.]],
         [[11., 12.],
          [0., 0.],
          [0., 0.],
          [0., 0.]]]
    ```

  Args:
    data: A tensor with shape `[D1, D2]`.
    sizes: An `int` tensor with shape `[A1, ..., An]`.
    max_rows: An `int` specifying the maximum number of rows in the unflattened
      output. `max_rows >= max(sizes)`.
    name: A name for this op. Defaults to 'utils_unflatten_2d_to_batch'.

  Returns:
    A tensor with shape `[A1, A2, ..., max_rows, D2]`.
  """
  with tf.name_scope(name):
    data = tf.convert_to_tensor(value=data)
    sizes = tf.convert_to_tensor(value=sizes)
    if max_rows is None:
      max_rows = tf.reduce_max(input_tensor=sizes)
    else:
      max_rows = tf.convert_to_tensor(value=max_rows)

    shape.check_static(tensor=data, tensor_name="data", has_rank=2)
    if not sizes.dtype.is_integer:
      raise TypeError("'sizes' must have an integer type.")

    mask = tf.sequence_mask(sizes, max_rows)
    mask_indices = tf.cast(tf.where(mask), tf.int32)
    output_shape = tf.concat(
        (tf.shape(input=sizes), (max_rows,), tf.shape(input=data)[-1:]), axis=0)
    return tf.scatter_nd(indices=mask_indices, updates=data, shape=output_shape)


def convert_to_block_diag_2d(data: tf.sparse.SparseTensor,
                             sizes: Optional[type_alias.TensorLike] = None,
                             validate_indices: bool = False,
                             name: str = "utils_convert_to_block_diag_2d"):
  """Convert a batch of 2d SparseTensors to a 2d block diagonal SparseTensor.

  Note:
    In the following, A1 to An are optional batch dimensions.

  A `SparseTensor` with dense shape `[A1, ..., An, D1, D2]` will be reshaped
  to one with shape `[A1*...*An*D1, A1*...*An*D2]`.

  Padded inputs in dims D1 and D2 are allowed. `sizes` indicates the un-padded
  shape for each inner `[D1, D2]` matrix. The additional (padded) rows and
  columns will be omitted in the block diagonal output.

  If padded (`sizes != None`), the input should not contain any sparse indices
  outside the bounds indicated by `sizes`. Setting `validate_indices=True` will
  explicitly filter any invalid sparse indices before block diagonalization.

  Args:
    data: A `SparseTensor` with dense shape `[A1, ..., An, D1, D2]`.
    sizes: A tensor with shape `[A1, ..., An, 2]`. Can be `None` (indicates no
      padding). If not `None`, `sizes` indicates the true sizes (before padding)
      of the inner dimensions of `data`.
    validate_indices: A boolean. Ignored if `sizes==None`. If True,
      out-of-bounds indices in `data` are explicitly ignored, otherwise
      out-of-bounds indices will cause undefined behavior.
    name: A name for this op. Defaults to 'utils_convert_to_block_diag_2d'.

  Returns:
    A 2d block-diagonal SparseTensor.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  with tf.name_scope(name):
    data = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=data)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)

    if not isinstance(data, tf.SparseTensor):
      raise TypeError("'data' must be a 'SparseTensor'.")
    if sizes is not None and not sizes.dtype.is_integer:
      raise TypeError("'sizes' must have an integer type.")
    shape.check_static(tensor=data, tensor_name="data", has_rank_greater_than=2)
    if sizes is not None:
      shape.check_static(
          tensor=sizes,
          tensor_name="sizes",
          has_rank=data.shape.ndims - 1,
          has_dim_equals=(-1, 2))
      shape.compare_batch_dimensions(
          tensors=(data, sizes),
          tensor_names=("data", "sizes"),
          last_axes=(-3, -2),
          broadcast_compatible=False)

    data_shape = tf.shape(input=data)
    data = tf.sparse.reshape(data, [-1, data_shape[-2], data_shape[-1]])
    indices = data.indices
    if sizes is not None:
      sizes = tf.cast(tf.reshape(sizes, shape=(-1, 2)), tf.int64)
      if validate_indices:
        in_bounds = ~tf.reduce_any(
            input_tensor=indices[:, 1:] >= tf.gather(sizes, indices[:, 0]),
            axis=-1)
        indices = tf.boolean_mask(tensor=indices, mask=in_bounds)
        values = tf.boolean_mask(tensor=data.values, mask=in_bounds)
      else:
        values = data.values
      cumsum = tf.cumsum(sizes, axis=0, exclusive=True)
      index_shift = tf.gather(cumsum, indices[:, 0])
      indices = indices[:, 1:] + index_shift
      block_diag = tf.SparseTensor(indices, values,
                                   tf.reduce_sum(input_tensor=sizes, axis=0))
    else:
      data_shape = tf.shape(input=data, out_type=tf.int64)
      index_shift = tf.expand_dims(indices[:, 0], -1) * data_shape[1:]
      indices = indices[:, 1:] + index_shift
      block_diag = tf.SparseTensor(indices, data.values,
                                   data_shape[0] * data_shape[1:])
    return block_diag


# API contains all public functions and classes.
__all__ = []
