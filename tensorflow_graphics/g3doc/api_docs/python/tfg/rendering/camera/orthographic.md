<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.camera.orthographic" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.rendering.camera.orthographic

This module implements orthographic camera functionalities.



Defined in [`rendering/camera/orthographic.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/camera/orthographic.py).

<!-- Placeholder for "Used in" -->

An orthographic camera represents three-dimensional objects in two dimensions
by parallel projection, in which the projection lines are parallel to the
camera axis. The camera axis is the line perpendicular to the image plane
starting at the camera center.

## Functions

[`project(...)`](../../../tfg/rendering/camera/orthographic/project.md): Projects a 3d point onto the 2d camera plane.

[`ray(...)`](../../../tfg/rendering/camera/orthographic/ray.md): Computes the 3d ray for a 2d point (the z component of the ray is 1).

[`unproject(...)`](../../../tfg/rendering/camera/orthographic/unproject.md): Unprojects a 2d point in 3d.

