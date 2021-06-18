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
"""This module implements various graph convolutions in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Dict
from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def feature_steered_convolution(
    data: type_alias.TensorLike,
    neighbors: tf.sparse.SparseTensor,
    sizes: type_alias.TensorLike,
    var_u: type_alias.TensorLike,
    var_v: type_alias.TensorLike,
    var_c: type_alias.TensorLike,
    var_w: type_alias.TensorLike,
    var_b: type_alias.TensorLike,
    name="graph_convolution_feature_steered_convolution") -> tf.Tensor:
  #  pyformat: disable
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
    name: A name for this op. Defaults to
      `graph_convolution_feature_steered_convolution`.

  Returns:
    Tensor with shape `[A1, ..., An, V, D]`.

  Raises:
    TypeError: if the input types are invalid.
    ValueError: if the input dimensions are invalid.
  """
  #  pyformat: enable
  with tf.name_scope(name):
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
    y_i_m = []
    x_sep = tf.gather(x_flat, adjacency_ind_1)
    q_m_list = tf.unstack(weights_q, axis=-1)
    w_m_list = tf.unstack(var_w, axis=0)

    x_flat_shape = tf.shape(input=x_flat)
    for q_m, w_m in zip(q_m_list, w_m_list):
      # Compute `y_i_m = sum_{j in neighborhood(i)} q_m(x_i, x_j) * w_m * x_j`.
      q_m = tf.expand_dims(q_m, axis=-1)
      p_sum = tf.math.unsorted_segment_sum(
          data=(q_m * x_sep) * tf.expand_dims(adjacency.values, -1),
          segment_ids=adjacency_ind_0,
          num_segments=x_flat_shape[0])
      y_i_m.append(tf.matmul(p_sum, w_m))
    y_out = tf.add_n(inputs=y_i_m) + tf.reshape(var_b, [1, -1])
    if data_ndims > 2:
      y_out = unflatten(y_out)
    return y_out


def edge_convolution_template(
    data: type_alias.TensorLike,
    neighbors: tf.sparse.SparseTensor,
    sizes: type_alias.TensorLike,
    edge_function: Callable[[type_alias.TensorLike, type_alias.TensorLike],
                            type_alias.TensorLike],
    reduction: str,
    edge_function_kwargs: Dict[str, Any],
    name: str = "graph_convolution_edge_convolution_template") -> tf.Tensor:
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
  with tf.name_scope(name):
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
      edge_features_weighted = edge_features * tf.expand_dims(
          adjacency.values, -1)
      features = tf.math.unsorted_segment_sum(
          data=edge_features_weighted,
          segment_ids=adjacency_ind_0,
          num_segments=tf.shape(input=x_flat)[0])
    elif reduction == "max":
      features = tf.math.segment_max(
          data=edge_features, segment_ids=adjacency_ind_0)
    else:
      raise ValueError("The reduction method must be 'weighted' or 'max'")

    features.set_shape(
        features.shape.merge_with(
            (tf.compat.dimension_value(x_flat.shape[0]),
             tf.compat.dimension_value(edge_features.shape[-1]))))

    if data_ndims > 2:
      features = unflatten(features)
    return features


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
