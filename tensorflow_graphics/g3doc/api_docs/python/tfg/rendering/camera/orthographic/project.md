<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.orthographic.project" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.orthographic.project

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/orthographic.py">View source</a>



Projects a 3d point onto the 2d camera plane.

```python
tfg.rendering.camera.orthographic.project(
    point_3d, name=None
)
```



<!-- Placeholder for "Used in" -->

Projects a 3d point \\((x, y, z)\\) to a 2d point \\((x', y')\\) onto the
image plane, with

$$
\begin{matrix}
x' = x, & y' = y.
\end{matrix}
$$

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_3d`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point to project.
* <b>`name`</b>: A name for this op that defaults to "orthographic_project".


#### Returns:

A tensor of shape `[A1, ..., An, 2]`, where the last dimension represents
a 2d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_3d` is not supported.