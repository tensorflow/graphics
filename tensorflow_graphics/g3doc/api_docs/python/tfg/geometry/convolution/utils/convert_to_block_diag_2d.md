<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.convolution.utils.convert_to_block_diag_2d" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.convolution.utils.convert_to_block_diag_2d

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/utils.py">View
source</a>

Convert a batch of 2d SparseTensors to a 2d block diagonal SparseTensor.

``` python
tfg.geometry.convolution.utils.convert_to_block_diag_2d(
    data,
    sizes=None,
    validate_indices=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

#### Note:

In the following, A1 to An are optional batch dimensions.

A `SparseTensor` with dense shape `[A1, ..., An, D1, D2]` will be reshaped
to one with shape `[A1*...*An*D1, A1*...*An*D2]`.

Padded inputs in dims D1 and D2 are allowed. `sizes` indicates the un-padded
shape for each inner `[D1, D2]` matrix. The additional (padded) rows and
columns will be omitted in the block diagonal output.

If padded (`sizes != None`), the input should not contain any sparse indices
outside the bounds indicated by `sizes`. Setting `validate_indices=True` will
explicitly filter any invalid sparse indices before block diagonalization.

#### Args:

*   <b>`data`</b>: A `SparseTensor` with dense shape `[A1, ..., An, D1, D2]`.
*   <b>`sizes`</b>: A tensor with shape `[A1, ..., An, 2]`. Can be `None`
    (indicates no padding). If not `None`, `sizes` indicates the true sizes
    (before padding) of the inner dimensions of `data`.
*   <b>`validate_indices`</b>: A boolean. Ignored if `sizes==None`. If True,
    out-of-bounds indices in `data` are explicitly ignored, otherwise
    out-of-bounds indices will cause undefined behavior.
*   <b>`name`</b>: A name for this op. Defaults to
    'utils_convert_to_block_diag_2d'.

#### Returns:

A 2d block-diagonal SparseTensor.

#### Raises:

* <b>`TypeError`</b>: if the input types are invalid.
* <b>`ValueError`</b>: if the input dimensions are invalid.