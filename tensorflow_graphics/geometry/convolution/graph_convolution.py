#Copyright 2018 Google LLC
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
"""This module implements various graph convolutions in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _prepare_feature_steered_args(
      data, neighbors, sizes, var_u, var_v, var_c, var_w, var_b):
  data = tf.convert_to_tensor(value=data)
  neighbors = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=neighbors)
  if sizes is not None:
    sizes = tf.convert_to_tensor(value=sizes)
  var_u = tf.convert_to_tensor(value=var_u)
  var_v = tf.convert_to_tensor(value=var_v)
  var_c = tf.convert_to_tensor(value=var_c)
  var_w = tf.convert_to_tensor(value=var_w)
  var_b = tf.convert_to_tensor(value=var_b)

  data_ndims = data.shape.ndims
  utils.check_valid_graph_convolution_input(data, neighbors, sizes)
  shape.compare_dimensions(
      tensors=(data, var_u, var_v, var_w),
      tensor_names=("data", "var_u", "var_v", "var_w"),
      axes=(-1, 0, 0, 1))
  shape.compare_dimensions(
      tensors=(var_u, var_v, var_c, var_w),
      tensor_names=("var_u", "var_v", "var_c", "var_w"),
      axes=(1, 1, 0, 0))
  shape.compare_dimensions(
      tensors=(var_w, var_b), tensor_names=("var_w", "var_b"), axes=-1)

  # Flatten the batch dimensions and remove any vertex padding.
  if data_ndims > 2:
    if sizes is not None:
      sizes_square = tf.stack((sizes, sizes), axis=-1)
    else:
      sizes_square = None
    x_flat, unflatten = utils.flatten_batch_to_2d(data, sizes)
    adjacency = utils.convert_to_block_diag_2d(neighbors, sizes_square)
  else:
    x_flat = data
    adjacency = neighbors
    unflatten = None

  return x_flat, adjacency, var_u, var_v, var_c, var_w, var_b, unflatten


def feature_steered_convolution_v1(data,
                                   neighbors,
                                   sizes,
                                   var_u,
                                   var_v,
                                   var_c,
                                   var_w,
                                   var_b,
                                   memory_efficient=True,
                                   segment_sum_impl='partition2d',
                                   name=None):
  """Implements the Feature Steered graph convolution.

  FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis
  Nitika Verma, Edmond Boyer, Jakob Verbeek
  CVPR 2018
  https://arxiv.org/abs/1706.05206

  Original implementation with some tweaks. Original version recovered with
  `memory_efficient=False, segment_sum_impl='partition2d'`.

  Additional args:
    memory_efficient: bool, if True uses `foldl` implementation which is
      slightly slower (~10% in experiments) but significantly more memory
      efficient (~2-4x less memory).
    segment_sum_impl: one of 'partition2d', 'sorted', 'unsorted', corresponding to
      using `tf.math.segment_sum`, `tf.math.unsorted_segment_sum` or
       `utils.partition_sums_2d` respectively. If 'sorted', `neighbors` must be
       ordered - see `tf.sparse.reorder`.
  """
  with tf.compat.v1.name_scope(
      name, "graph_convolution_feature_steered_convolution",
      [data, neighbors, sizes, var_u, var_v, var_c, var_w, var_b]):
    x_flat, adjacency, var_u, var_v, var_c, var_w, var_b, unflatten = \
        _prepare_feature_steered_args(data, neighbors, sizes, var_u, var_v,
                                      var_c, var_w, var_b)
    x_u = tf.matmul(x_flat, var_u)
    x_v = tf.matmul(x_flat, var_v)
    adjacency_ind_0 = adjacency.indices[:, 0]
    adjacency_ind_1 = adjacency.indices[:, 1]
    x_u_rep = tf.gather(x_u, adjacency_ind_0)
    x_v_sep = tf.gather(x_v, adjacency_ind_1)
    weights_q = tf.exp(x_u_rep + x_v_sep + tf.reshape(var_c, (1, -1)))
    weights_q_sum = tf.reduce_sum(
        input_tensor=weights_q, axis=-1, keepdims=True)
    weights_q = weights_q / weights_q_sum
    x_sep = tf.gather(x_flat, adjacency_ind_1)
    V = tf.shape(x_flat)[0]

    def get_mth_term(q_m, w_m):
      if segment_sum_impl == 'partition2d':
        q_m = tf.expand_dims(q_m, axis=-1)
        p_sum = utils.partition_sums_2d(q_m * x_sep, adjacency_ind_0,
                                        adjacency.values)
      else:
        args = (x_sep * tf.expand_dims(q_m * adjacency.values, axis=-1),
                adjacency_ind_0)
        if segment_sum_impl == 'sorted':
          p_sum = tf.math.segment_sum(*args)
        elif segment_sum_impl == 'unsorted':
          p_sum = tf.math.unsorted_segment_sum(*args, num_segments=V)
        else:
          raise ValueError(
              'Invalid segment_sum_impl "{}" - must be one of "partition2d", '
              '"sorted", "unsorted"'.format(segment_sum_impl))
      return tf.matmul(p_sum, w_m)

    if memory_efficient:
      y_out = tf.foldl(
          lambda acc, args: acc + get_mth_term(*args),
          (tf.transpose(weights_q, (1, 0)), var_w),
          tf.tile(tf.expand_dims(var_b, axis=0), (tf.shape(x_flat)[0], 1)))
    else:
      q_ms = tf.unstack(weights_q, axis=-1)
      w_ms = tf.unstack(var_w, axis=0)
      y_out = tf.add_n(
          [get_mth_term(*args) for args in zip(q_ms, w_ms)]) + var_b

    if unflatten is not None:
      y_out = unflatten(y_out)
    return y_out


def feature_steered_convolution_v2(data,
                                   neighbors,
                                   sizes,
                                   var_u,
                                   var_v,
                                   var_c,
                                   var_w,
                                   var_b,
                                   transform_data_first=None,
                                   name=None):
  """Implements the Feature Steered graph convolution.

  FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis
  Nitika Verma, Edmond Boyer, Jakob Verbeek
  CVPR 2018
  https://arxiv.org/abs/1706.05206

  This implementation is based on splitting the exponential term in the softmax
  into products of expoentials which allows neighborhood summation to be
  implemented as a sparse-dense matrix product. This means per-edge features
  (other than the softmax values) need not be explicitly created, so memory
  usage is lower and computation is faster.

  Extra channels ("W", or "M" in paper) are broadcast to create a large feature
  matrix of shape [V, D, M] before reduction. This is slightly faster at the
  cost of a larger memory footprint. See `feature_steered_convolution_v3` for
  a slightly slower more memory efficient implementation.

  For base arg/return descriptions, see See `feature_steered_convolution`.

  Additional args:
    transform_data_first: if True, performs transformation of features from
      [V, C] -> [V, D, W] via `var_w` before other multiplications.
      Defaults to `C > D`.
  """
  with tf.compat.v1.name_scope(
      name, "graph_convolution_feature_steered_convolution_v2",
      [data, neighbors, sizes, var_u, var_v, var_c, var_w, var_b]):
    x_flat, adjacency, var_u, var_v, var_c, var_w, var_b, unflatten = \
        _prepare_feature_steered_args(data, neighbors, sizes, var_u, var_v,
                                      var_c, var_w, var_b)
    x_u = tf.matmul(x_flat, var_u)  # [V, W]
    x_v = tf.matmul(x_flat, var_v)  # [V, W]
    x_uc = x_u + var_c  # [V, W]
    # apply per-term stabilization
    x_uc = x_uc - tf.reduce_max(x_uc, axis=-1, keepdims=True)
    x_v = x_v - tf.reduce_max(x_v, axis=-1, keepdims=True)

    e_uc = tf.exp(x_uc)
    e_v = tf.exp(x_v)

    i, j = tf.unstack(adjacency.indices, axis=-1)
    # E == num_edges
    q_vals = tf.gather(e_uc, i) * tf.gather(e_v, j)  # [E, W]
    weights = adjacency.values / tf.reduce_sum(q_vals, axis=-1)  # [E]

    weighted_adjacency = tf.SparseTensor(
      adjacency.indices, weights, dense_shape=adjacency.dense_shape)

    # `tf.einsum` implementations arguable easier to understand and possibly
    # more efficient, but we avoid them until the following issue is resolved
    # https://github.com/tensorflow/tensorflow/issues/31022
    # Seems to be limited to examples where indices are repeated but not summed?

    W, C, D = var_w.shape
    assert(C is not None and D is not None)
    if transform_data_first is None:
      transform_data_first = C > D

    if transform_data_first:
      # x_flat = tf.einsum('vc,wcd->vdw', x_flat, var_w)
      x_flat = tf.reduce_sum(tf.multiply(
          tf.reshape(x_flat, (-1, 1, C, 1)),  # V 1 C W
          tf.transpose(var_w, (2, 1, 0)),       # D C W
        ), axis=2)                            # V D   W
      F = D
    else:
      F = C
      x_flat = tf.expand_dims(x_flat, axis=-1)  # V C 1

    # ef = tf.einsum('vw,vfw->vfw', e_v, data)
    ef = tf.expand_dims(e_v, axis=-2) * x_flat  # [V, F, W]
    ef = tf.reshape(ef, (-1, F * W))            # [V, F * W]
    summed_ef = tf.sparse.sparse_dense_matmul(weighted_adjacency, ef)
    summed_ef = tf.reshape(summed_ef, (-1, F, W))  # [V, F, W]

    if transform_data_first:
      # ym = tf.einsum('vfw,vw->vf', summed_ef, e_uc)
      ym_flat = summed_ef * tf.expand_dims(e_uc, axis=1)
      y_flat = tf.reduce_sum(ym_flat, axis=-1)
    else:
      # y_flat = tf.einsum('vfw,vw,wcd->vd', summed_ef, e_uc, var_w)
      ym_flat = summed_ef * tf.expand_dims(e_uc, axis=1)
      y_flat = tf.matmul(
        tf.reshape(ym_flat, (-1, C * W)),
        tf.reshape(tf.transpose(var_w, (1, 0, 2)), (C * W, D))
      )
    y_flat = y_flat + var_b
    if unflatten is not None:
      return unflatten(y_flat)
    else:
      return y_flat


def feature_steered_convolution_v3(data,
                                   neighbors,
                                   sizes,
                                   var_u,
                                   var_v,
                                   var_c,
                                   var_w,
                                   var_b,
                                   memory_efficient=True,
                                   name=None):
  """Implements the Feature Steered graph convolution.

  FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis
  Nitika Verma, Edmond Boyer, Jakob Verbeek
  CVPR 2018
  https://arxiv.org/abs/1706.05206

  This implementation is similar to `feature_steered_convolution_v2` except
  it loops over entries of `var_w` in feature transformation. This avoids the
  need to have a single [V, D, F] tensor in memory at any point in time.

  For base arg/return descriptions, see See `feature_steered_convolution`.

  Additional args:
    memory_efficient: bool, if True uses `foldl` implementation which is
      slightly slower (~10% in experiments) but significantly more memory
      efficient (~2-4x less memory).
  """
  with tf.compat.v1.name_scope(
      name, "graph_convolution_feature_steered_convolution",
      [data, neighbors, sizes, var_u, var_v, var_c, var_w, var_b]):
    x_flat, adjacency, var_u, var_v, var_c, var_w, var_b, unflatten = \
        _prepare_feature_steered_args(data, neighbors, sizes, var_u, var_v,
                                      var_c, var_w, var_b)
    x_u = tf.matmul(x_flat, var_u)  # [V, W]
    x_v = tf.matmul(x_flat, var_v)  # [V, W]
    x_uc = x_u + var_c  # [V, W]

    # apply per-term stabilization
    x_uc = x_uc - tf.reduce_max(x_uc, axis=-1, keepdims=True)
    x_v = x_v - tf.reduce_max(x_v, axis=-1, keepdims=True)

    e_uc = tf.exp(x_uc)
    e_v = tf.exp(x_v)

    i, j = tf.unstack(adjacency.indices, axis=-1)
    # E == num_edges
    q_vals = tf.gather(e_uc, i) * tf.gather(e_v, j)  # [E, W]
    weights = adjacency.values / tf.reduce_sum(q_vals, axis=-1)  # [E]

    weighted_adjacency = tf.SparseTensor(
      adjacency.indices, weights, dense_shape=adjacency.dense_shape)

    W, C, D = var_w.shape
    assert(C is not None and D is not None)

    def get_mth_term(wm, e_ucm, e_vm):
      summed_ef = tf.sparse.sparse_dense_matmul(
        weighted_adjacency, tf.expand_dims(e_vm, axis=-1) * x_flat)
      return tf.matmul(tf.expand_dims(e_ucm, axis=-1) * summed_ef, wm)

    if memory_efficient:
      y_flat = tf.foldl(
          lambda acc, args: acc + get_mth_term(*args),
          (var_w, tf.transpose(e_uc, (1, 0)), tf.transpose(e_v, (1, 0))),
          tf.tile(tf.expand_dims(var_b, axis=0), (tf.shape(e_uc)[0], 1)))
    else:
      args = [
        tf.unstack(var_w, axis=0),
        tf.unstack(e_uc, axis=1),
        tf.unstack(e_v, axis=1),
      ]
      y_flat = tf.add_n([get_mth_term(*args) for args in zip(*args)]) + var_b

    if unflatten is not None:
      return unflatten(y_flat)
    else:
      return y_flat


_versions = {
  'v1': feature_steered_convolution_v1,
  'v2': feature_steered_convolution_v2,
  'v3': feature_steered_convolution_v3,
}


def feature_steered_convolution(data,
                                neighbors,
                                sizes,
                                var_u,
                                var_v,
                                var_c,
                                var_w,
                                var_b,
                                version='v1',
                                name=None,
                                **kwargs):
  """Implements the Feature Steered graph convolution.

  FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis
  Nitika Verma, Edmond Boyer, Jakob Verbeek
  CVPR 2018
  https://arxiv.org/abs/1706.05206

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.
    `D`: The number of channels in the output after convolution.
    `W`: The number of weight matrices used in the convolution.
    The input variables (`var_u`, `var_v`, `var_c`, `var_w`, `var_b`) correspond
    to the variables with the same names in the paper cited above.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., An, V, V]` representing vertex neighborhoods. The neighborhood
      of a vertex defines the support region for convolution. For a mesh, a
      common choice for the neighborhood of vertex i would be the vertices in
      the K-ring of i (including i itself). Each vertex must have at least one
      neighbor. For a faithful implementation of the FeaStNet convolution,
      neighbors should be a row-normalized weight matrix corresponding to the
      graph adjacency matrix with self-edges: `neighbors[A1, ..., An, i, j] > 0`
      if vertex j is a neighbor of i, and `neighbors[A1, ..., An, i, i] > 0` for
      all i, and `sum(neighbors, axis=-1)[A1, ..., An, i] == 1.0 for all i`.
      These requirements are relaxed in this implementation.
    sizes: An `int` tensor of shape `[A1, ..., An]` indicating the true input
      sizes in case of padding (`sizes=None` indicates no padding).Note that
      `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. An example usage of `sizes`: consider an input consisting of
      three graphs G0, G1, and G2 with V0, V1, and V2 vertices respectively. The
      padded input would have the following shapes: `data.shape = [3, V, C]` and
      `neighbors.shape = [3, V, V]`, where `V = max([V0, V1, V2])`. The true
      sizes of each graph will be specified by `sizes=[V0, V1, V2]`,
      `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]` will be the vertex and
      neighborhood data of graph Gi. The `SparseTensor` `neighbors` should have
      no nonzero entries in the padded regions.
    var_u: A 2-D tensor with shape `[C, W]`.
    var_v: A 2-D tensor with shape `[C, W]`.
    var_c: A 1-D tensor with shape `[W]`.
    var_w: A 3-D tensor with shape `[W, C, D]`.
    var_b: A 1-D tensor with shape `[D]`.
    version: string indicating implementation version, one of "v1", "v2", "v3".
      See `feature_steered_convolution_v1` / `feature_steered_convolution_v2`
        etc.
    name: A name for this op. Defaults to
      `graph_convolution_feature_steered_convolution`.
    **kwargs: version-specific kwargs.
        use_original_segment_sum (default False): for "v1", use the original
          implementation of segment sum, rather than
          `tf.math.unsorted_segment_sum`.
        memory_efficient (default True): for "v1", "v3", uses a more memory
          efficient implementation at the cost of slightly slower runtime.
        transform_data_first: for "v2", if True transforms data from
          [V, C] -> [V, D, W] before some transformations. This does not affect
          the results (aside from floating point errors), but may result in
          a performance difference. Defaults to `C > D`.

  Returns:
    Tensor with shape `[A1, ..., An, V, D]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  if version not in _versions:
    raise ValueError(
      'Invalid version {}. Must be one of {}'.format(
        version, sorted(_versions)))
  return _versions[version](
    data,
    neighbors,
    sizes,
    var_u,
    var_v,
    var_c,
    var_w,
    var_b,
    name=name,
    **kwargs)


def edge_convolution_template(data,
                              neighbors,
                              sizes,
                              edge_function,
                              reduction,
                              edge_function_kwargs,
                              name=None):
  #  pyformat: disable
  r"""A template for edge convolutions.

  This function implements a general edge convolution for graphs of the form
  \\(y_i = \sum_{j \in \mathcal{N}(i)} w_{ij} f(x_i, x_j)\\), where
  \\(\mathcal{N}(i)\\) is the set of vertices in the neighborhood of vertex
  \\(i\\), \\(x_i \in \mathbb{R}^C\\) are the features at vertex \\(i\\),
  \\(w_{ij} \in \mathbb{R}\\) is the weight for the edge between vertex \\(i\\)
  and vertex \\(j\\), and finally
  \\(f(x_i, x_j): \mathbb{R}^{C} \times \mathbb{R}^{C} \to \mathbb{R}^{D}\\) is
  a user-supplied function.

  This template also implements the same general edge convolution described
  above with a max-reduction instead of a weighted sum.

  An example of how this template can be used is for Laplacian smoothing,
  which is defined as
  $$y_i = \frac{1}{|\mathcal{N(i)}|} \sum_{j \in \mathcal{N(i)}} x_j$$.
  `edge_convolution_template` can be used to perform Laplacian smoothing by
  setting
  $$w_{ij} = \frac{1}{|\mathcal{N(i)}|}$$, `edge_function=lambda x, y: y`,
  and `reduction='weighted'`.

  The shorthands used below are
    `V`: The number of vertices.
    `C`: The number of channels in the input data.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    data: A `float` tensor with shape `[A1, ..., An, V, C]`.
    neighbors: A `SparseTensor` with the same type as `data` and with shape
      `[A1, ..., An, V, V]` representing vertex neighborhoods. The neighborhood
      of a vertex defines the support region for convolution. The value at
      `neighbors[A1, ..., An, i, j]` corresponds to the weight \\(w_{ij}\\)
      above. Each vertex must have at least one neighbor.
    sizes: An `int` tensor of shape `[A1, ..., An]` indicating the true input
      sizes in case of padding (`sizes=None` indicates no padding). Note that
      `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
      be ignored. As an example, consider an input consisting of three graphs
      G0, G1, and G2 with V0, V1, and V2 vertices respectively. The padded input
      would have the shapes `[3, V, C]`, and `[3, V, V]` for `data` and
      `neighbors` respectively, where `V = max([V0, V1, V2])`. The true sizes of
      each graph will be specified by `sizes=[V0, V1, V2]` and `data[i, :Vi, :]`
      and `neighbors[i, :Vi, :Vi]` will be the vertex and neighborhood data of
      graph Gi. The `SparseTensor` `neighbors` should have no nonzero entries in
      the padded regions.
    edge_function: A callable that takes at least two arguments of vertex
      features and returns a tensor of vertex features. `Y = f(X1, X2,
      **kwargs)`, where `X1` and `X2` have shape `[V3, C]` and `Y` must have
      shape `[V3, D], D >= 1`.
    reduction: Either 'weighted' or 'max'. Specifies the reduction over the
        neighborhood. For 'weighted', the reduction is a weighted sum as shown
        in the equation above. For 'max' the reduction is a max over features in
        which case the weights $$w_{ij}$$ are ignored.
    edge_function_kwargs: A dict containing any additional keyword arguments to
      be passed to `edge_function`.
    name: A name for this op. Defaults to
      `graph_convolution_edge_convolution_template`.

  Returns:
    Tensor with shape `[A1, ..., An, V, D]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  #  pyformat: enable
  with tf.compat.v1.name_scope(name,
                               "graph_convolution_edge_convolution_template",
                               [data, neighbors, sizes]):
    data = tf.convert_to_tensor(value=data)
    neighbors = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=neighbors)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)

    data_ndims = data.shape.ndims
    utils.check_valid_graph_convolution_input(data, neighbors, sizes)

    # Flatten the batch dimensions and remove any vertex padding.
    if data_ndims > 2:
      if sizes is not None:
        sizes_square = tf.stack((sizes, sizes), axis=-1)
      else:
        sizes_square = None
      x_flat, unflatten = utils.flatten_batch_to_2d(data, sizes)
      adjacency = utils.convert_to_block_diag_2d(neighbors, sizes_square)
    else:
      x_flat = data
      adjacency = neighbors

    adjacency_ind_0 = adjacency.indices[:, 0]
    adjacency_ind_1 = adjacency.indices[:, 1]
    vertex_features = tf.gather(x_flat, adjacency_ind_0)
    neighbor_features = tf.gather(x_flat, adjacency_ind_1)
    edge_features = edge_function(vertex_features, neighbor_features,
                                  **edge_function_kwargs)

    if reduction == "weighted":
      features = utils.partition_sums_2d(edge_features, adjacency_ind_0,
                                         adjacency.values)
    elif reduction == "max":
      features = tf.math.segment_max(data=edge_features,
                                     segment_ids=adjacency_ind_0)
      features.set_shape(features.shape.merge_with(
          (tf.compat.v1.dimension_value(x_flat.shape[0]),
           tf.compat.v1.dimension_value(edge_features.shape[-1]))))
    else:
      raise ValueError("The reduction method must be 'weighted' or 'max'")

    if data_ndims > 2:
      features = unflatten(features)
    return features


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
