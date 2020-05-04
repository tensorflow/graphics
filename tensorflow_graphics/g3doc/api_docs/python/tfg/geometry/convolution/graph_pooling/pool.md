<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.graph_pooling.pool" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.graph_pooling.pool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_pooling.py">View source</a>



Implements graph pooling.

```python
tfg.geometry.convolution.graph_pooling.pool(
    data, pool_map, sizes, algorithm='max', name=None
)
```



<!-- Placeholder for "Used in" -->

The features at each output vertex are computed by pooling over a subset of
vertices in the input graph. This pooling window is specified by the input
`pool_map`.

The shorthands used below are
  `V1`: The number of vertices in the input data.
  `V2`: The number of vertices in the pooled output data.
  `C`: The number of channels in the data.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`data`</b>: A `float` tensor with shape `[A1, ..., An, V1, C]`.
* <b>`pool_map`</b>: A `SparseTensor` with the same type as `data` and with shape
  `[A1, ..., An, V2, V1]`. The features for an output vertex `v2` will be
  computed by pooling over the corresponding input vertices specified by
  the entries in `pool_map[A1, ..., An, v2, :]`.
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., An, 2]` indicating the true
  input sizes in case of padding (`sizes=None` indicates no padding).
  `sizes[A1, ..., An, 0] <= V2` specifies the padding in the (pooled)
  output, and `sizes[A1, ..., An, 1] <= V1` specifies the padding in the
  input.
* <b>`algorithm`</b>: The pooling function, must be either 'max' or 'weighted'. Default
  is 'max'. For 'max' pooling, the output features are the maximum over the
  input vertices (in this case only the indices of the `SparseTensor`
  `pool_map` are used, the values are ignored). For 'weighted', the output
  features are a weighted sum of the input vertices, the weights specified
  by the values of `pool_map`.
* <b>`name`</b>: A name for this op. Defaults to 'graph_pooling_pool'.


#### Returns:

Tensor with shape `[A1, ..., An, V2, C]`.



#### Raises:


* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.
* <b>`ValueError`</b>: if `algorithm` is invalid.