<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.graph_pooling.upsample_transposed_convolution" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.graph_pooling.upsample_transposed_convolution

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_pooling.py">View source</a>



Graph upsampling by transposed convolution.

```python
tfg.geometry.convolution.graph_pooling.upsample_transposed_convolution(
    data, pool_map, sizes, kernel_size, transposed_convolution_op, name=None
)
```



<!-- Placeholder for "Used in" -->

Upsamples a graph using a transposed convolution op. The map from input
vertices to the upsampled graph is specified by the reverse of pool_map. The
inputs `pool_map` and `sizes` are the same as used for pooling:

```
>>> pooled = pool(data, pool_map, sizes)
>>> upsampled = upsample_transposed_convolution(pooled, pool_map, sizes, ...)
```

The shorthands used below are
  `V1`: The number of vertices in the inputs.
  `V2`: The number of vertices in the upsampled output.
  `C`: The number of channels in the inputs.

#### Note:

In the following, A1 to A3 are optional batch dimensions. Only up to three
batch dimensions are supported due to limitations with TensorFlow's
dense-sparse multiplication.


Please see the documentation for <a href="../../../../tfg/geometry/convolution/graph_pooling/pool.md"><code>graph_pooling.pool</code></a> for a detailed
interpretation of the inputs `pool_map` and `sizes`.

#### Args:


* <b>`data`</b>: A `float` tensor with shape `[A1, ..., A3, V1, C]`.
* <b>`pool_map`</b>: A `SparseTensor` with the same type as `data` and with shape
  `[A1, ..., A3, V1, V2]`. `pool_map` will be interpreted in the same way
  as the `pool_map` argument of <a href="../../../../tfg/geometry/convolution/graph_pooling/pool.md"><code>graph_pooling.pool</code></a>, namely
  `v_i_map = [..., v_i, :]` are the upsampled vertices corresponding to
  vertex `v_i`. Additionally, for transposed convolution a fixed number of
  entries in each `v_i_map` (equal to `kernel_size`) are expected:
  `|v_i_map| = kernel_size`. When this is not the case, the map is either
  truncated or the last element repeated. Furthermore, upsampled vertex
  indices should not be repeated across maps otherwise the output is
  nondeterministic. Specifically, to avoid nondeterminism we must have
  `intersect([a1, ..., an, v_i, :],[a1, ..., a3, v_j, :]) = {}, i != j`.
* <b>`sizes`</b>: An `int` tensor of shape `[A1, ..., A3, 2]` indicating the true
  input sizes in case of padding (`sizes=None` indicates no padding):
  `sizes[A1, ..., A3, 0] <= V1` and `sizes[A1, ..., A3, 1] <= V2`.
* <b>`kernel_size`</b>: The kernel size for transposed convolution.
* <b>`transposed_convolution_op`</b>: A callable transposed convolution op with the
  form `y = transposed_convolution_op(x)`, where `x` has shape
  `[1, 1, D1, C]` and `y` must have shape `[1, 1, kernel_size * D1, C]`.
  `transposed_convolution_op` maps each row of `x` to `kernel_size` rows
  in `y`. An example:
  `transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
      filters=C, kernel_size=(1, kernel_size), strides=(1, kernel_size),
      padding='valid', ...)
* <b>`name`</b>: A name for this op. Defaults to
  'graph_pooling_upsample_transposed_convolution'.


#### Returns:

Tensor with shape `[A1, ..., A3, V2, C]`.



#### Raises:


* <b>`TypeError`</b>: if the input types are invalid.
* <b>`TypeError`</b>: if `transposed_convolution_op` is not a callable.
* <b>`ValueError`</b>: if the input dimensions are invalid.