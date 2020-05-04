<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.nn.loss.chamfer_distance.evaluate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.nn.loss.chamfer_distance.evaluate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py">View source</a>



Computes the Chamfer distance for the given two point sets.

```python
tfg.nn.loss.chamfer_distance.evaluate(
    point_set_a, point_set_b, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

This is a symmetric version of the Chamfer distance, calculated as the sum
of the average minimum distance from point_set_a to point_set_b and vice
versa.
The average minimum distance from one point set to another is calculated as
the average of the distances between the points in the first set and their
closest point in the second set, and is thus not symmetrical.



#### Note:

This function returns the exact Chamfer distance and not an approximation.



#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`point_set_a`</b>: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
  represents points in a D dimensional space.
* <b>`point_set_b`</b>: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
  represents points in a D dimensional space.
* <b>`name`</b>: A name for this op. Defaults to "chamfer_distance_evaluate".


#### Returns:

A tensor of shape `[A1, ..., An]` storing the chamfer distance between the
two point sets.



#### Raises:


* <b>`ValueError`</b>: if the shape of `point_set_a`, `point_set_b` is not supported.