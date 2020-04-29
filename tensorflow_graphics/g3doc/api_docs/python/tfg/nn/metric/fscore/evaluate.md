<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.nn.metric.fscore.evaluate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.nn.metric.fscore.evaluate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/metric/fscore.py">View source</a>



Computes the fscore metric for the given ground truth and predicted labels.

```python
tfg.nn.metric.fscore.evaluate(
    ground_truth, prediction, precision_function=tfg.nn.metric.precision.evaluate,
    recall_function=tfg.nn.metric.recall.evaluate, name=None
)
```



<!-- Placeholder for "Used in" -->

The fscore is calculated as 2 * (precision * recall) / (precision + recall)
where the precision and recall are evaluated by the given function parameters.
The precision and recall functions default to their definition for boolean
labels (see https://en.wikipedia.org/wiki/Precision_and_recall for more
details).

#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`ground_truth`</b>: A tensor of shape `[A1, ..., An, N]`, where the last axis
  represents the ground truth values.
* <b>`prediction`</b>: A tensor of shape `[A1, ..., An, N]`, where the last axis
  represents the predicted values.
* <b>`precision_function`</b>: The function to use for evaluating the precision.
  Defaults to the precision evaluation for binary ground-truth and
  predictions.
* <b>`recall_function`</b>: The function to use for evaluating the recall. Defaults to
  the recall evaluation for binary ground-truth and prediction.
* <b>`name`</b>: A name for this op. Defaults to "fscore_evaluate".


#### Returns:

A tensor of shape `[A1, ..., An]` that stores the fscore metric for the
given ground truth labels and predictions.



#### Raises:


* <b>`ValueError`</b>: if the shape of `ground_truth`, `prediction` is
not supported.