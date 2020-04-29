<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.nn.metric.intersection_over_union.evaluate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.nn.metric.intersection_over_union.evaluate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/metric/intersection_over_union.py">View source</a>



Computes the Intersection-Over-Union metric for the given ground truth and predicted labels.

```python
tfg.nn.metric.intersection_over_union.evaluate(
    ground_truth_labels, predicted_labels, grid_size=1, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible, and G1 to Gm are the grid dimensions.



#### Args:


* <b>`ground_truth_labels`</b>: A tensor of shape `[A1, ..., An, G1, ..., Gm]`, where
  the last m axes represent a grid of ground truth attributes. Each
  attribute can either be 0 or 1.
* <b>`predicted_labels`</b>: A tensor of shape `[A1, ..., An, G1, ..., Gm]`, where the
  last m axes represent a grid of predicted attributes. Each attribute can
  either be 0 or 1.
* <b>`grid_size`</b>: The number of grid dimensions. Defaults to 1.
* <b>`name`</b>: A name for this op. Defaults to "intersection_over_union_evaluate".


#### Returns:

A tensor of shape `[A1, ..., An]` that stores the intersection-over-union
metric of the given ground truth labels and predictions.



#### Raises:


* <b>`ValueError`</b>: if the shape of `ground_truth_labels`, `predicted_labels` is
not supported.