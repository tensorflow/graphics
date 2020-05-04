<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective.project" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.perspective.project

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



Projects a 3d point onto the 2d camera plane.

```python
tfg.rendering.camera.perspective.project(
    point_3d, focal, principal_point, name=None
)
```



<!-- Placeholder for "Used in" -->

Projects a 3d point \\((x, y, z)\\) to a 2d point \\((x', y')\\) onto the
image plane with

$$
\begin{matrix}
x' = \frac{f_x}{z}x + c_x, & y' = \frac{f_y}{z}y + c_y,
\end{matrix}
$$

where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
point.

#### Note:

In the following, A1 to An are optional batch dimensions that must be
broadcast compatible.



#### Args:


* <b>`point_3d`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point to project.
* <b>`focal`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a camera focal length.
* <b>`principal_point`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension represents a camera principal point.
* <b>`name`</b>: A name for this op that defaults to "perspective_project".


#### Returns:

A tensor of shape `[A1, ..., An, 2]`, where the last dimension represents
a 2d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_3d`, `focal`, or `principal_point` is not
supported.