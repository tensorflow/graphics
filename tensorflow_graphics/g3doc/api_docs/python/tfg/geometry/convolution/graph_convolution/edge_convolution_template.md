<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.graph_convolution.edge_convolution_template" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.graph_convolution.edge_convolution_template

A template for edge convolutions.

``` python
tfg.geometry.convolution.graph_convolution.edge_convolution_template(
    data,
    neighbors,
    sizes,
    edge_function,
    edge_function_kwargs,
    name=None
)
```



Defined in [`geometry/convolution/graph_convolution.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_convolution.py).

<!-- Placeholder for "Used in" -->

This function implements a general edge convolution for graphs of the form
\\(y_i = \sum_{j \in \mathcal{N}(i)} w_{ij} f(x_i, x_j)\\), where
\\(\mathcal{N}(i)\\) is the set of vertices in the neighborhood of vertex
\\(i\\), \\(x_i \in \mathbb{R}^C\\) are the features at vertex \\(i\\),
\\(w_{ij} \in \mathbb{R}\\) is the weight for the edge between vertex \\(i\\)
and vertex \\(j\\), and finally
\\(f(x_i, x_j): \mathbb{R}^{C} \times \mathbb{R}^{C} \to \mathbb{R}^{D}\\) is
a user-supplied function.

The shorthands used below are
  `V`: The number of vertices.
  `C`: The number of channels in the input data.

#### Note:

In the following, A1 to An are optional batch dimensions.


#### Args:

* <b>`data`</b>: A `float` tensor with shape `[A1, ..., An, V, C]`.
* <b>`neighbors`</b>: A `SparseTensor` with the same type as `data` and with shape
  `[A1, ..., An, V, V]` representing vertex neighborhoods. The neighborhood
  of a vertex defines the support region for convolution. The value at
  `neighbors[A1, ..., An, i, j]` corresponds to the weight \\(w_{ij}\\)
  above. Each vertex must have at least one neighbor.
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., An]` indicating the true input
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
* <b>`edge_function`</b>: A callable that takes at least two arguments of vertex
  features and returns a tensor of vertex features. `Y = f(X1, X2,
  **kwargs)`, where `X1` and `X2` have shape `[V3, C]` and `Y` must have
  shape `[V3, D], D >= 1`.
* <b>`edge_function_kwargs`</b>: A dict containing any additional keyword arguments to
  be passed to `edge_function`.
* <b>`name`</b>: A name for this op. Defaults to
  `graph_convolution_edge_convolution_template`.


#### Returns:

Tensor with shape `[A1, ..., An, V, D]`.


#### Raises:

* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.