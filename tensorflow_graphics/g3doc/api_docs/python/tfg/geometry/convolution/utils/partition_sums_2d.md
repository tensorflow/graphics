<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils.partition_sums_2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.utils.partition_sums_2d

Sum over subsets of rows in a 2-D tensor.

``` python
tfg.geometry.convolution.utils.partition_sums_2d(
    data,
    group_ids,
    row_weights=None,
    name=None
)
```



Defined in [`geometry/convolution/utils.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`data`</b>: 2-D tensor with shape `[D1, D2]`.
* <b>`group_ids`</b>: 1-D `int` tensor with shape `[D1]`.
* <b>`row_weights`</b>: 1-D tensor with shape `[D1]`. Can be `None`.
* <b>`name`</b>: A name for this op. Defaults to `utils_partition_sums_2d`.


#### Returns:

A 2-D tensor with shape `[max(group_ids) + 1, D2]` where
  `output[i, :] = sum(data[j, :] * weight[j] * 1(group_ids[j] == i)),
  1(.) is the indicator function.


#### Raises:

* <b>`ValueError`</b>: if the inputs have invalid dimensions or types.