<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.slerp.interpolate_with_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.slerp.interpolate_with_weights

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/slerp.py">View source</a>



Interpolates vectors by taking their weighted sum.

```python
tfg.math.interpolation.slerp.interpolate_with_weights(
    vector1, vector2, weight1, weight2, name=None
)
```



<!-- Placeholder for "Used in" -->

Interpolation for all variants of slerp is a simple weighted sum over inputs.
Therefore this function simply returns weight1 * vector1 + weight2 * vector2.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vector1`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`vector2`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`weight1`</b>: A `float` or a tensor describing weights for the `vector1` and with
  a shape broadcastable to the shape of the input vectors.
* <b>`weight2`</b>: A `float` or a tensor describing weights for the `vector2` and with
  a shape broadcastable to the shape of the input vectors.
* <b>`name`</b>: A name for this op. Defaults to "interpolate_with_weights".


#### Returns:

A tensor of shape `[A1, ... , An, M]` containing the result of the
interpolation.
