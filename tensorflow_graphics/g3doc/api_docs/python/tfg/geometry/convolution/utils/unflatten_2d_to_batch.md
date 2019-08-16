<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils.unflatten_2d_to_batch" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.utils.unflatten_2d_to_batch

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py">View
source</a>

Reshapes a 2d Tensor into a batch of 2d Tensors.

``` python
tfg.geometry.convolution.utils.unflatten_2d_to_batch(
    data,
    sizes,
    max_rows=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The `data` tensor with shape `[D1, D2]` will be mapped to a tensor with shape
`[A1, ..., An, max_rows, D2]` where `max_rows` defaults to `max(sizes)`.
`sizes` determines the segment of rows in the input that get mapped to a
particular batch dimension (`sum(sizes) == D1`).

#### Examples:


```python
data = [[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.],
        [9., 10.],
        [11., 12.]]
sizes = [2, 3, 1]

output = unflatten_2d_to_batch(data, sizes, max_rows=None)
print(output.shape)
>>> [3, 3, 2]
print(output)
>>> [[[1., 2.],
      [3., 4.],
      [0., 0.]],
     [[5., 6.],
      [7., 8.],
      [9., 10.]],
     [[11., 12.],
      [0., 0.],
      [0., 0.]]]

output = unflatten_2d_to_batch(data, sizes, max_rows=4)
print(output.shape)
>>> [3, 4, 2]
print(output)
>>> [[[1., 2.],
      [3., 4.],
      [0., 0.],
      [0., 0.]],
     [[5., 6.],
      [7., 8.],
      [9., 10.],
      [0., 0.]],
     [[11., 12.],
      [0., 0.],
      [0., 0.],
      [0., 0.]]]
```

#### Args:

*   <b>`data`</b>: A tensor with shape `[D1, D2]`.
*   <b>`sizes`</b>: An `int` tensor with shape `[A1, ..., An]`.
*   <b>`max_rows`</b>: An `int` specifying the maximum number of rows in the
    unflattened output. `max_rows >= max(sizes)`.
*   <b>`name`</b>: A name for this op. Defaults to
    'utils_unflatten_2d_to_batch'.

#### Returns:

A tensor with shape `[A1, A2, ..., max_rows, D2]`.
