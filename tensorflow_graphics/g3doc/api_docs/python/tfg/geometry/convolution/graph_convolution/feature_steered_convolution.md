<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.graph_convolution.feature_steered_convolution" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.graph_convolution.feature_steered_convolution

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_convolution.py">View source</a>



Implements the Feature Steered graph convolution.

```python
tfg.geometry.convolution.graph_convolution.feature_steered_convolution(
    data, neighbors, sizes, var_u, var_v, var_c, var_w, var_b, name=None
)
```



<!-- Placeholder for "Used in" -->

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

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`data`</b>: A `float` tensor with shape `[A1, ..., An, V, C]`.
* <b>`neighbors`</b>: A `SparseTensor` with the same type as `data` and with shape
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
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., An]` indicating the true input
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
* <b>`var_u`</b>: A 2-D tensor with shape `[C, W]`.
* <b>`var_v`</b>: A 2-D tensor with shape `[C, W]`.
* <b>`var_c`</b>: A 1-D tensor with shape `[W]`.
* <b>`var_w`</b>: A 3-D tensor with shape `[W, C, D]`.
* <b>`var_b`</b>: A 1-D tensor with shape `[D]`.
* <b>`name`</b>: A name for this op. Defaults to
  `graph_convolution_feature_steered_convolution`.


#### Returns:

Tensor with shape `[A1, ..., An, V, D]`.



#### Raises:


* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.