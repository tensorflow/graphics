<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.rotate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.rotate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.

```python
tfg.geometry.transformation.axis_angle.rotate(
    point, axis, angle, name=None
)
```



<!-- Placeholder for "Used in" -->

Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
$$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:

$$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
+\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point to rotate.
* <b>`axis`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a normalized axis.
* <b>`angle`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents an angle.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_rotate".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d point.



#### Raises:


* <b>`ValueError`</b>: If `point`, `axis`, or `angle` are of different shape or if
their respective shape is not supported.