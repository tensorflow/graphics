<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.orthographic.ray" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.camera.orthographic.ray

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/orthographic.py">View source</a>



Computes the 3d ray for a 2d point (the z component of the ray is 1).

```python
tfg.rendering.camera.orthographic.ray(
    point_2d, name=None
)
```



<!-- Placeholder for "Used in" -->

Computes the 3d ray \\((r_x, r_y, 1)\\) for a 2d point \\((x', y')\\) on the
image plane. For an orthographic camera the rays are constant over the image
plane with

$$
\begin{matrix}
r_x = 0, & r_y = 0, & z = 1.
\end{matrix}
$$

Note: In the following, A1 to An are optional batch dimensions.

#### Args:


* <b>`point_2d`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last dimension
  represents a 2d point.
* <b>`name`</b>: A name for this op that defaults to "orthographic_ray".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d ray.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point_2d` is not supported.