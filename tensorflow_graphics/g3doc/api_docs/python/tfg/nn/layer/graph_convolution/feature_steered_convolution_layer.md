<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.nn.layer.graph_convolution.feature_steered_convolution_layer" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.nn.layer.graph_convolution.feature_steered_convolution_layer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/layer/graph_convolution.py">View source</a>



Wraps the function `feature_steered_convolution` as a TensorFlow layer.

```python
tfg.nn.layer.graph_convolution.feature_steered_convolution_layer(
    data, neighbors, sizes, translation_invariant=True, num_weight_matrices=8,
    num_output_channels=None,
    initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1), name=None,
    var_name=None
)
```



<!-- Placeholder for "Used in" -->

The shorthands used below are
  `V`: The number of vertices.
  `C`: The number of channels in the input data.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`data`</b>: A `float` tensor with shape `[A1, ..., An, V, C]`.
* <b>`neighbors`</b>: A SparseTensor with the same type as `data` and with shape
  `[A1, ..., An, V, V]` representing vertex neighborhoods. The neighborhood
  of a vertex defines the support region for convolution. For a mesh, a
  common choice for the neighborhood of vertex `i` would be the vertices in
  the K-ring of `i` (including `i` itself). Each vertex must have at least
  one neighbor. For a faithful implementation of the FeaStNet paper,
  neighbors should be a row-normalized weight matrix corresponding to the
  graph adjacency matrix with self-edges:
  `neighbors[A1, ..., An, i, j] > 0` if vertex `i` and `j` are neighbors,
  `neighbors[A1, ..., An, i, i] > 0` for all `i`, and
  `sum(neighbors, axis=-1)[A1, ..., An, i] == 1.0` for all `i`.
  These requirements are relaxed in this implementation.
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., An]` indicating the true input
  sizes in case of padding (`sizes=None` indicates no padding).
  `sizes[A1, ..., An] <= V`. If `data` and `neighbors` are 2-D, `sizes` will
  be ignored. As an example, consider an input consisting of three graphs
  `G0`, `G1`, and `G2` with `V0`, `V1` and `V2` vertices respectively. The
  padded input would have the following shapes: `data.shape = [3, V, C]`,
  and `neighbors.shape = [3, V, V]`, where `V = max([V0, V1, V2])`. The true
  sizes of each graph will be specified by `sizes=[V0, V1, V2]`.
  `data[i, :Vi, :]` and `neighbors[i, :Vi, :Vi]` will be the vertex and
  neighborhood data of graph `Gi`. The `SparseTensor` `neighbors` should
  have no nonzero entries in the padded regions.
* <b>`translation_invariant`</b>: A `bool`. If `True` the assignment of features to
  weight matrices will be invariant to translation.
* <b>`num_weight_matrices`</b>: An `int` specifying the number of weight matrices used
  in the convolution.
* <b>`num_output_channels`</b>: An optional `int` specifying the number of channels in
  the output. If `None` then `num_output_channels = C`.
* <b>`initializer`</b>: An initializer for the trainable variables.
* <b>`name`</b>: A (name_scope) name for this op. Passed through to
  feature_steered_convolution().
* <b>`var_name`</b>: A (var_scope) name for the variables. Defaults to
  `graph_convolution_feature_steered_convolution_weights`.


#### Returns:

Tensor with shape `[A1, ..., An, V, num_output_channels]`.
