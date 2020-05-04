<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective.ray" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.perspective.ray

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



Computes the 3d ray for a 2d point (the z component of the ray is 1).

```python
tfg.rendering.camera.perspective.ray(
    point_2d, focal, principal_point, name=None
)
```



<!-- Placeholder for "Used in" -->

Computes the 3d ray \\((r_x, r_y, 1)\\) from the camera center to a 2d point
\\((x', y')\\) on the image plane with

$$
\begin{matrix}
r_x = \frac{(x' - c_x)}{f_x}, & r_y = \frac{(y' - c_y)}{f_y}, & z = 1,
\end{matrix}
$$

where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
point. The camera optical center is assumed to be at \\((0, 0, 0)\\).

#### Note:

In the following, A1 to An are optional batch dimensions that must be
broadcast compatible.



#### Args:


* <b>`point_2d`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a 2d point.
* <b>`focal`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a camera focal length.
* <b>`principal_point`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension represents a camera principal point.
* <b>`name`</b>: A name for this op that defaults to "perspective_ray".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d ray.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_2d`, `focal`, or `principal_point` is not
supported.