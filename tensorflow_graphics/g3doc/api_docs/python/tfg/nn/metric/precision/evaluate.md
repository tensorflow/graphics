<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.nn.metric.precision.evaluate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.nn.metric.precision.evaluate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/metric/precision.py">View source</a>



Computes the precision metric for the given ground truth and predictions.

```python
tfg.nn.metric.precision.evaluate(
    ground_truth, prediction, classes=None, reduce_average=True,
    prediction_to_category_function=_cast_to_int, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`ground_truth`</b>: A tensor of shape `[A1, ..., An, N]`, where the last axis
  represents the ground truth labels. Will be cast to int32.
* <b>`prediction`</b>: A tensor of shape `[A1, ..., An, N]`, where the last axis
  represents the predictions (which can be continuous).
* <b>`classes`</b>: An integer or a list/tuple of integers representing the classes for
  which the precision will be evaluated. In case 'classes' is 'None', the
  number of classes will be inferred from the given labels and the precision
  will be calculated for each of the classes. Defaults to 'None'.
* <b>`reduce_average`</b>: Whether to calculate the average of the precision for each
  class and return a single precision value. Defaults to true.
* <b>`prediction_to_category_function`</b>: A function to associate a `prediction` to a
  category. Defaults to rounding down the value of the prediction to the
  nearest integer value.
* <b>`name`</b>: A name for this op. Defaults to "precision_evaluate".


#### Returns:

A tensor of shape `[A1, ..., An, C]`, where the last axis represents the
precision calculated for each of the requested classes.



#### Raises:


* <b>`ValueError`</b>: if the shape of `ground_truth`, `prediction` is not supported.