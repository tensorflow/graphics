<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils.check_valid_graph_unpooling_input" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.utils.check_valid_graph_unpooling_input

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py">View
source</a>

Checks that the inputs are valid for graph unpooling.

``` python
tfg.geometry.convolution.utils.check_valid_graph_unpooling_input(
    data,
    pool_map,
    sizes
)
```



<!-- Placeholder for "Used in" -->

#### Note:

In the following, A1 to A3 are optional batch dimensions.

#### Args:

*   <b>`data`</b>: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
*   <b>`pool_map`</b>: A `SparseTensor` with the same type as `data` and with
    shape `[A1, ..., A3, V1, V2]`.
*   <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., A3, 2]`. Can be `None`.

#### Raises:

* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.