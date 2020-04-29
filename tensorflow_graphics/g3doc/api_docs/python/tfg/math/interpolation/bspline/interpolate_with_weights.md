<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.bspline.interpolate_with_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.bspline.interpolate_with_weights

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/bspline.py">View source</a>



Interpolates knots using knot weights.

```python
tfg.math.interpolation.bspline.interpolate_with_weights(
    knots, weights, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An, and B1 to Bk are optional batch dimensions.



#### Args:


* <b>`knots`</b>: A tensor with shape `[B1, ..., Bk, C]` containing knot values, where
  `C` is the number of knots.
* <b>`weights`</b>: A tensor with shape `[A1, ..., An, C]` containing dense weights for
  the knots, where `C` is the number of knots.
* <b>`name`</b>: A name for this op. Defaults to "bspline_interpolate_with_weights".


#### Returns:

A tensor with shape `[A1, ..., An, B1, ..., Bk]`, which is the result of
spline interpolation.



#### Raises:


* <b>`ValueError`</b>: If the last dimension of knots and weights is not equal.