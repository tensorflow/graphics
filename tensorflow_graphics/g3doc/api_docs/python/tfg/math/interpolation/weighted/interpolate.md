<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.weighted.interpolate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.weighted.interpolate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/weighted.py">View source</a>



Weighted interpolation for M-D point sets.

```python
tfg.math.interpolation.weighted.interpolate(
    points, weights, indices, normalize=True, allow_negative_weights=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Given an M-D point set, this function can be used to generate a new point set
that is formed by interpolating a subset of points in the set.

#### Note:

In the following, A1 to An, and B1 to Bk are optional batch dimensions.



#### Args:


* <b>`points`</b>: A tensor with shape `[B1, ..., Bk, M] and rank R > 1, where M is the
  dimensionality of the points.
* <b>`weights`</b>: A tensor with shape `[A1, ..., An, P]`, where P is the number of
  points to interpolate for each output point.
* <b>`indices`</b>: A tensor of dtype tf.int32 and shape `[A1, ..., An, P, R-1]`, which
  contains the point indices to be used for each output point. The R-1
  dimensional axis gives the slice index of a single point in `points`. The
  first n+1 dimensions of weights and indices must match, or be broadcast
  compatible.
* <b>`normalize`</b>: A `bool` describing whether or not to normalize the weights on
  the last axis.
* <b>`allow_negative_weights`</b>: A `bool` describing whether or not negative weights
  are allowed.
* <b>`name`</b>: A name for this op. Defaults to "weighted_interpolate".


#### Returns:

A tensor of shape `[A1, ..., An, M]` storing the interpolated M-D
points. The first n dimensions will be the same as weights and indices.
