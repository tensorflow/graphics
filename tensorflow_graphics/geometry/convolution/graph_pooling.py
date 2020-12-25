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
"""This module implements various graph pooling ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import export_api


def pool(data, pool_map, sizes, algorithm='max', name=None):
  #  pyformat: disable
  """Implements graph pooling.

  The features at each output vertex are computed by pooling over a subset of
  vertices in the input graph. This pooling window is specified by the input
  `pool_map`.

  The shorthands used below are
    `V1`: The number of vertices in the input data.
    `V2`: The number of vertices in the pooled output data.
    `C`: The number of channels in the data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V1, C]`.
    pool_map: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., An, V2, V1]`. The features for an output vertex `v2` will be
      computed by pooling over the corresponding input vertices specified by
      the entries in `pool_map[A1, ..., An, v2, :]`.
    sizes: An `int` tensor of shape `[A1, ..., An, 2]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding).
      `sizes[A1, ..., An, 0] <= V2` specifies the padding in the (pooled)
      output, and `sizes[A1, ..., An, 1] <= V1` specifies the padding in the
      input.
    algorithm: The pooling function, must be either 'max' or 'weighted'. Default
      is 'max'. For 'max' pooling, the output features are the maximum over the
      input vertices (in this case only the indices of the `SparseTensor`
      `pool_map` are used, the values are ignored). For 'weighted', the output
      features are a weighted sum of the input vertices, the weights specified
      by the values of `pool_map`.
    name: A name for this op. Defaults to 'graph_pooling_pool'.

  Returns:
    Tensor with shape `[A1, ..., An, V2, C]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
    ValueError: if `algorithm` is invalid.
  """
  #  pyformat: enable
  with tf.compat.v1.name_scope(
      name, 'graph_pooling_pool', [data, pool_map, sizes]):
    data = tf.convert_to_tensor(value=data)
    pool_map = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=pool_map)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
    utils.check_valid_graph_pooling_input(data, pool_map, sizes)

    if sizes is not None:
      sizes_output, sizes_input = tf.split(sizes, 2, axis=-1)
      sizes_output = tf.squeeze(sizes_output, axis=-1)
      sizes_input = tf.squeeze(sizes_input, axis=-1)
    else:
      sizes_output = None
      sizes_input = None

    batched = data.shape.ndims > 2
    if batched:
      x_flat, _ = utils.flatten_batch_to_2d(data, sizes_input)
      pool_map_block_diagonal = utils.convert_to_block_diag_2d(pool_map, sizes)
    else:
      x_flat = data
      pool_map_block_diagonal = pool_map

    if algorithm == 'weighted':
      pooled = tf.sparse.sparse_dense_matmul(pool_map_block_diagonal, x_flat)
    elif algorithm == 'max':
      pool_groups = tf.gather(x_flat, pool_map_block_diagonal.indices[:, 1])
      pooled = tf.math.segment_max(
          data=pool_groups, segment_ids=pool_map_block_diagonal.indices[:, 0])
    else:
      raise ValueError('The pooling method must be "weighted" or "max"')

    if batched:
      if sizes_output is not None:
        pooled = utils.unflatten_2d_to_batch(pooled, sizes_output)
      else:
        output_shape = tf.concat((tf.shape(input=pool_map)[:-1], (-1,)), axis=0)
        pooled = tf.reshape(pooled, output_shape)

    return pooled


def unpool(data, pool_map, sizes, name=None):
  #  pyformat: disable
  r"""Graph upsampling by inverting the pooling map.

  Upsamples a graph by applying a pooling map in reverse. The inputs `pool_map`
  and `sizes` are the same as used for pooling:

  >>> pooled = pool(data, pool_map, sizes)
  >>> upsampled = unpool(pooled, pool_map, sizes)

  The shorthands used below are
    `V1`: The number of vertices in the input data.
    `V2`: The number of vertices in the unpooled output data.
    `C`: The number of channels in the data.

  Note:
    In the following, A1 to A3 are optional batch dimensions. Only up to three
    batch dimensions are supported due to limitations with TensorFlow's
    dense-sparse multiplication.

  Please see the documentation for `graph_pooling.pool` for a detailed
  interpretation of the inputs `pool_map` and `sizes`.

  Args:
    data: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
    pool_map: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., A3, V1, V2]`. The features for vertex `v1` are computed by
      pooling over the entries in `pool_map[A1, ..., A3, v1, :]`. This function
      applies this pooling map in reverse.
    sizes: An `int` tensor of shape `[A1, ..., A3, 2]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding):
      `sizes[A1, ..., A3, 0] <= V1` and `sizes[A1, ..., A3, 1] <= V2`.
    name: A name for this op. Defaults to 'graph_pooling_unpool'.

  Returns:
    Tensor with shape `[A1, ..., A3, V2, C]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  #  pyformat: enable
  with tf.compat.v1.name_scope(
      name, 'graph_pooling_unpool', [data, pool_map, sizes]):
    data = tf.convert_to_tensor(value=data)
    pool_map = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=pool_map)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)
    utils.check_valid_graph_unpooling_input(data, pool_map, sizes)

    # Reverse pool_map and sizes.
    pool_map_ndims = pool_map.shape.ndims
    permutation = tf.concat((tf.range(pool_map_ndims - 2),
                             (pool_map_ndims - 1, pool_map_ndims - 2)),
                            axis=0)
    pool_map_transpose = tf.sparse.transpose(pool_map, permutation)
    row_sum = tf.sparse.reduce_sum(
        tf.abs(pool_map_transpose), keepdims=True, axis=-1)
    normalize_weights = tf.compat.v1.where(
        tf.equal(row_sum, 0), row_sum, 1.0 / row_sum)
    pool_map_normalize = normalize_weights * pool_map_transpose

    if sizes is not None:
      sizes = tf.reverse(sizes, axis=(-1,))

    return pool(data, pool_map_normalize, sizes)


def upsample_transposed_convolution(data,
                                    pool_map,
                                    sizes,
                                    kernel_size,
                                    transposed_convolution_op,
                                    name=None):
  #  pyformat: disable
  r"""Graph upsampling by transposed convolution.

  Upsamples a graph using a transposed convolution op. The map from input
  vertices to the upsampled graph is specified by the reverse of pool_map. The
  inputs `pool_map` and `sizes` are the same as used for pooling:

  >>> pooled = pool(data, pool_map, sizes)
  >>> upsampled = upsample_transposed_convolution(pooled, pool_map, sizes, ...)

  The shorthands used below are
    `V1`: The number of vertices in the inputs.
    `V2`: The number of vertices in the upsampled output.
    `C`: The number of channels in the inputs.

  Note:
    In the following, A1 to A3 are optional batch dimensions. Only up to three
    batch dimensions are supported due to limitations with TensorFlow's
    dense-sparse multiplication.

  Please see the documentation for `graph_pooling.pool` for a detailed
  interpretation of the inputs `pool_map` and `sizes`.

  Args:
    data: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
    pool_map: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., A3, V1, V2]`. `pool_map` will be interpreted in the same way
      as the `pool_map` argument of `graph_pooling.pool`, namely
      `v_i_map = [..., v_i, :]` are the upsampled vertices corresponding to
      vertex `v_i`. Additionally, for transposed convolution a fixed number of
      entries in each `v_i_map` (equal to `kernel_size`) are expected:
      `|v_i_map| = kernel_size`. When this is not the case, the map is either
      truncated or the last element repeated. Furthermore, upsampled vertex
      indices should not be repeated across maps otherwise the output is
      nondeterministic. Specifically, to avoid nondeterminism we must have
      `intersect([a1, ..., an, v_i, :],[a1, ..., a3, v_j, :]) = {}, i != j`.
    sizes: An `int` tensor of shape `[A1, ..., A3, 2]` indicating the true
      input sizes in case of padding (`sizes=None` indicates no padding):
      `sizes[A1, ..., A3, 0] <= V1` and `sizes[A1, ..., A3, 1] <= V2`.
    kernel_size: The kernel size for transposed convolution.
    transposed_convolution_op: A callable transposed convolution op with the
      form `y = transposed_convolution_op(x)`, where `x` has shape
      `[1, 1, D1, C]` and `y` must have shape `[1, 1, kernel_size * D1, C]`.
      `transposed_convolution_op` maps each row of `x` to `kernel_size` rows
      in `y`. An example:
      `transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
          filters=C, kernel_size=(1, kernel_size), strides=(1, kernel_size),
          padding='valid', ...)
    name: A name for this op. Defaults to
      'graph_pooling_upsample_transposed_convolution'.

  Returns:
    Tensor with shape `[A1, ..., A3, V2, C]`.

  Raises:
    TypeError: if the input types are invalid.
    TypeError: if `transposed_convolution_op` is not a callable.
    ValueError: if the input dimensions are invalid.
  """
  #  pyformat: enable
  with tf.compat.v1.name_scope(
      name, 'graph_pooling_upsample_transposed_convolution',
      [data, pool_map, sizes]):
    data = tf.convert_to_tensor(value=data)
    pool_map = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=pool_map)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)

    utils.check_valid_graph_unpooling_input(data, pool_map, sizes)
    if not callable(transposed_convolution_op):
      raise TypeError("'transposed_convolution_op' must be callable.")

    if sizes is not None:
      sizes_input, sizes_output = tf.split(sizes, 2, axis=-1)
      sizes_input = tf.squeeze(sizes_input, axis=-1)
      sizes_output = tf.squeeze(sizes_output, axis=-1)
    else:
      sizes_input = None
      sizes_output = None

    num_features = tf.compat.v1.dimension_value(data.shape[-1])
    batched = data.shape.ndims > 2
    if batched:
      x_flat, _ = utils.flatten_batch_to_2d(data, sizes_input)
      pool_map_block_diagonal = utils.convert_to_block_diag_2d(pool_map, sizes)
    else:
      x_flat = data
      pool_map_block_diagonal = pool_map

    x_flat = tf.expand_dims(tf.expand_dims(x_flat, 0), 0)
    x_upsample = transposed_convolution_op(x_flat)

    # Map each upsampled vertex into its correct position based on pool_map.
    # Select 'kernel_size' neighbors for each input vertex. Truncate or repeat
    # as necessary.
    ragged = tf.RaggedTensor.from_value_rowids(
        pool_map_block_diagonal.indices[:, 1],
        pool_map_block_diagonal.indices[:, 0])
    # Take up to the first 'kernel_size' entries.
    ragged_k = ragged[:, :kernel_size]
    # Fill rows with less than 'kernel_size' entries by repeating the last
    # entry.
    last = ragged_k[:, -1:].flat_values
    num_repeat = kernel_size - ragged_k.row_lengths()
    sum_num_repeat = tf.reduce_sum(input_tensor=num_repeat)
    ones_ragged = tf.RaggedTensor.from_row_lengths(
        tf.ones((sum_num_repeat,), dtype=last.dtype), num_repeat)
    repeat = ones_ragged * tf.expand_dims(last, -1)
    padded = tf.concat([ragged_k, repeat], axis=1)
    pool_map_dense = tf.reshape(padded.flat_values, (-1, kernel_size))

    # Map rows of 'x_upsample' to positions indicated by the
    # indices 'pool_map_dense'.
    up_scatter_indices = tf.expand_dims(tf.reshape(pool_map_dense, (-1,)), -1)
    up_row = tf.reshape(tf.cast(up_scatter_indices, tf.int64), (-1,))
    up_column = tf.range(tf.shape(input=up_row, out_type=tf.dtypes.int64)[0])
    scatter_indices = tf.concat(
        (tf.expand_dims(up_row, -1),
         tf.expand_dims(up_column, -1)), axis=1)

    scatter_values = tf.ones_like(up_row, dtype=x_upsample.dtype)
    scatter_shape = tf.reduce_max(input_tensor=scatter_indices, axis=0) + 1

    scatter = tf.SparseTensor(scatter_indices,
                              scatter_values,
                              scatter_shape)
    scatter = tf.sparse.reorder(scatter)
    row_sum = tf.sparse.reduce_sum(tf.abs(scatter), keepdims=True, axis=-1)
    row_sum = tf.compat.v1.where(tf.equal(row_sum, 0.), row_sum, 1.0 / row_sum)
    scatter = row_sum * scatter
    x_upsample = tf.sparse.sparse_dense_matmul(scatter, x_upsample[0, 0, :, :])

    if batched:
      if sizes_output is not None:
        x_upsample = utils.unflatten_2d_to_batch(x_upsample, sizes_output)
      else:
        output_shape = tf.concat((tf.shape(input=pool_map)[:-2],
                                  tf.shape(input=pool_map)[-1:],
                                  (num_features,)), axis=0)
        x_upsample = tf.reshape(x_upsample, output_shape)

    return x_upsample

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
