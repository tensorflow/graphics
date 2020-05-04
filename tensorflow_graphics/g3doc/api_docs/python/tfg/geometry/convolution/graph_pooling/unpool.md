<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.graph_pooling.unpool" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.graph_pooling.unpool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_pooling.py">View source</a>



Graph upsampling by inverting the pooling map.

```python
tfg.geometry.convolution.graph_pooling.unpool(
    data, pool_map, sizes, name=None
)
```



<!-- Placeholder for "Used in" -->

Upsamples a graph by applying a pooling map in reverse. The inputs `pool_map`
and `sizes` are the same as used for pooling:

```
>>> pooled = pool(data, pool_map, sizes)
>>> upsampled = unpool(pooled, pool_map, sizes)
```

The shorthands used below are
  `V1`: The number of vertices in the input data.
  `V2`: The number of vertices in the unpooled output data.
  `C`: The number of channels in the data.

#### Note:

In the following, A1 to A3 are optional batch dimensions. Only up to three
batch dimensions are supported due to limitations with TensorFlow's
dense-sparse multiplication.


Please see the documentation for <a href="../../../../tfg/geometry/convolution/graph_pooling/pool.md"><code>graph_pooling.pool</code></a> for a detailed
interpretation of the inputs `pool_map` and `sizes`.

#### Args:


* <b>`data`</b>: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
* <b>`pool_map`</b>: A `SparseTensor` with the same type as `data` and with shape
  `[A1, ..., A3, V1, V2]`. The features for vertex `v1` are computed by
  pooling over the entries in `pool_map[A1, ..., A3, v1, :]`. This function
  applies this pooling map in reverse.
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., A3, 2]` indicating the true
  input sizes in case of padding (`sizes=None` indicates no padding):
  `sizes[A1, ..., A3, 0] <= V1` and `sizes[A1, ..., A3, 1] <= V2`.
* <b>`name`</b>: A name for this op. Defaults to 'graph_pooling_unpool'.


#### Returns:

Tensor with shape `[A1, ..., A3, V2, C]`.



#### Raises:


* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.