<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.slerp.vector_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.slerp.vector_weights

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/slerp.py">View source</a>



Spherical linear interpolation (slerp) between two unnormalized vectors.

```python
tfg.math.interpolation.slerp.vector_weights(
    vector1, vector2, percent, eps=None, name=None
)
```



<!-- Placeholder for "Used in" -->

This function applies geometric slerp to unnormalized vectors by first
normalizing them to return the interpolation weights. It reduces to lerp when
input vectors are exactly anti-parallel.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vector1`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`vector2`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`percent`</b>: A `float` or tensor with shape broadcastable to the shape of input
  vectors.
* <b>`eps`</b>: A small float for operation safety. If left None, its value is
  automatically selected using dtype of input vectors.
* <b>`name`</b>: A name for this op. Defaults to "vector_weights".


#### Raises:


* <b>`ValueError`</b>: if the shape of `vector1`, `vector2`, or `percent` is not
  supported.


#### Returns:

Two tensors of shape `[A1, ... , An, 1]`, representing interpolation weights
for each input vector.
