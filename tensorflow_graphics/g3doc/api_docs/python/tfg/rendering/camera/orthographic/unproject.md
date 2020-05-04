<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.orthographic.unproject" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.orthographic.unproject

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/orthographic.py">View source</a>



Unprojects a 2d point in 3d.

```python
tfg.rendering.camera.orthographic.unproject(
    point_2d, depth, name=None
)
```



<!-- Placeholder for "Used in" -->

Unprojects a 2d point \\((x', y')\\) to a 3d point \\((x, y, z)\\) given its
depth \\(z\\), with

$$
\begin{matrix}
x = x', & y = y', & z = z.
\end{matrix}
$$

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_2d`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a 2d point to unproject.
* <b>`depth`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents the depth of a 2d point.
* <b>`name`</b>: A name for this op that defaults to "orthographic_unproject".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_2d`, `depth` is not supported.