<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.perspective.unproject" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.perspective.unproject

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/perspective.py">View source</a>



Unprojects a 2d point in 3d.

```python
tfg.rendering.camera.perspective.unproject(
    point_2d, depth, focal, principal_point, name=None
)
```



<!-- Placeholder for "Used in" -->

Unprojects a 2d point \\((x', y')\\) to a 3d point \\((x, y, z)\\) knowing the
depth \\(z\\) with

$$
\begin{matrix}
x = \frac{z (x' - c_x)}{f_x}, & y = \frac{z(y' - c_y)}{f_y}, & z = z,
\end{matrix}
$$

where \\((f_x, f_y)\\) is the focal length and \\((c_x, c_y)\\) the principal
point.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_2d`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a 2d point to unproject.
* <b>`depth`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents the depth of a 2d point.
* <b>`focal`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a camera focal length.
* <b>`principal_point`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension represents a camera principal point.
* <b>`name`</b>: A name for this op that defaults to "perspective_unproject".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_2d`, `depth`, `focal`, or
`principal_point` is not supported.