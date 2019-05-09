<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils.flatten_batch_to_2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.utils.flatten_batch_to_2d

Reshapes a batch of 2d Tensors by flattening across the batch dimensions.

``` python
tfg.geometry.convolution.utils.flatten_batch_to_2d(
    data,
    sizes=None,
    name=None
)
```



Defined in [`geometry/convolution/utils.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py).

<!-- Placeholder for "Used in" -->

#### Note:

  In the following, A1 to An are optional batch dimensions.

A tensor with shape `[A1, ..., An, D1, D2]` will be reshaped to one
with shape `[A1*...*An*D1, D2]`. This function also returns an inverse
function that returns any tensor with shape `[A1*...*An*D1, D3]` to one
with shape `[A1, ..., An, D1, D3]`.

Padded inputs in dim D1 are allowed. `sizes` determines the first elements
from D1 to select from each batch dimension.


#### Examples:

```python
data = [[[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]],
        [[9., 10.], [11., 12.]]]
sizes = None
output = flatten_batch_to_2d(data, size)
print(output)
>>> [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]]

data = [[[1., 2.], [0., 0.]],
        [[5., 6.], [7., 8.]],
        [[9., 10.], [0., 0.]]]
sizes = [1, 2, 1]
output = flatten_batch_to_2d(data, size)
print(output)
>>> [[1., 2.], [5., 6.], [7., 8.], [9., 10.]]
```


#### Args:

* <b>`data`</b>: A tensor with shape `[A1, ..., An, D1, D2]`.
* <b>`sizes`</b>: An `int` tensor with shape `[A1, ..., An]`. Can be `None`. `sizes[i]
  <= D1`.
* <b>`name`</b>: A name for this op. Defaults to `utils_flatten_batch_to_2d`.


#### Returns:

A tensor with shape `[A1*...*An*D1, D2]` if `sizes == None`, otherwise a
  tensor  with shape `[sum(sizes), D2]`.
A function that reshapes a tensor with shape `[A1*...*An*D1, D3]` to a
  tensor with shape `[A1, ..., An, D1, D3]` if `sizes == None`, otherwise
  it reshapes a tensor with shape `[sum(sizes), D3]` to one with shape
  `[A1, ..., An, ..., D1, D3]`.


#### Raises:

* <b>`ValueError`</b>: if the input tensor dimensions are invalid.