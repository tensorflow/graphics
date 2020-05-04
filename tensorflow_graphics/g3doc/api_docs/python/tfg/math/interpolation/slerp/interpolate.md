<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.slerp.interpolate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.slerp.interpolate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/slerp.py">View source</a>



Applies slerp to vectors or quaternions.

```python
tfg.math.interpolation.slerp.interpolate(
    vector1, vector2, percent,
    method=tfg.math.interpolation.slerp.InterpolationType.QUATERNION, eps=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`vector1`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`vector2`</b>: A tensor of shape `[A1, ... , An, M]`, which stores a normalized
  vector in its last dimension.
* <b>`percent`</b>: A `float` or a tensor with shape broadcastable to the shape of
  input vectors.
* <b>`method`</b>: An enumerated constant from the class InterpolationType, which is
  either InterpolationType.QUATERNION (default) if the input vectors are 4-D
  quaternions, or InterpolationType.VECTOR if they are regular M-D vectors.
* <b>`eps`</b>: A small float for operation safety. If left None, its value is
  automatically selected using dtype of input vectors.
* <b>`name`</b>: A name for this op. Defaults to "vector_weights" or
  "quaternion_weights" depending on the method.


#### Returns:

A tensor of shape [A1, ... , An, M]` which stores the result of the
interpolation.



#### Raises:


* <b>`ValueError`</b>: if method is not amongst enumerated constants defined in
  InterpolationType.