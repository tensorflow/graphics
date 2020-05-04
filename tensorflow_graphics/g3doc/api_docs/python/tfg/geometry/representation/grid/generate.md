<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.grid.generate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.grid.generate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/grid.py">View source</a>



Generates a M-D uniform axis-aligned grid.

```python
tfg.geometry.representation.grid.generate(
    starts, stops, nums, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Warning:

This op is not differentiable. Indeed, the gradient of tf.linspace and
tf.meshgrid are currently not defined.



#### Note:

In the following, `B` is an optional batch dimension.



#### Args:


* <b>`starts`</b>: A tensor of shape `[M]` or `[B, M]`, where the last dimension
  represents a M-D start point.
* <b>`stops`</b>: A tensor of shape `[M]` or `[B, M]`, where the last dimension
  represents a M-D end point.
* <b>`nums`</b>: A tensor of shape `[M]` representing the number of subdivisions for
  each dimension.
* <b>`name`</b>: A name for this op. Defaults to "grid_generate".


#### Returns:

A tensor of shape `[nums[0], ..., nums[M-1], M]` containing an M-D uniform
  grid or a tensor of shape [B, nums[0], ..., nums[M-1], M]` containing B
  M-D uniform grids. Please refer to the example below for more details.



#### Raises:


* <b>`ValueError`</b>: If the shape of `starts`, `stops`, or 'nums' is not supported.


#### Examples:

```python
print(generate((-1.0, -2.0), (1.0, 2.0), (3, 5)))
>>> [[[-1. -2.]
      [-1. -1.]
      [-1.  0.]
      [-1.  1.]
      [-1.  2.]]
     [[ 0. -2.]
      [ 0. -1.]
      [ 0.  0.]
      [ 0.  1.]
      [ 0.  2.]]
     [[ 1. -2.]
      [ 1. -1.]
      [ 1.  0.]
      [ 1.  1.]
      [ 1.  2.]]]
```
Generates a 3x5 2d grid from -1.0 to 1.0 with 3 subdivisions for the x
axis and from -2.0 to 2.0 with 5 subdivisions for the y axis. This lead to a
tensor of shape (3, 5, 2).
