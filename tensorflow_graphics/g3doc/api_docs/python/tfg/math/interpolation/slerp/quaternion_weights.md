<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.slerp.quaternion_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.slerp.quaternion_weights

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/slerp.py">View source</a>



Calculates slerp weights for two normalized quaternions.

```python
tfg.math.interpolation.slerp.quaternion_weights(
    quaternion1, quaternion2, percent, eps=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Given a percent and two normalized quaternions, this function returns the
slerp weights. It can also produce extrapolation weights when percent is
outside of the [0, 1] range. It reduces to lerp when input quaternions are
almost parallel or anti-parallel. Input quaternions are assumed to be
normalized. The tf.graphics debug flag TFG_ADD_ASSERTS_TO_GRAPH defined
in tfg_flags.py can be set to add assertions to the graph that check whether
the inputs are normalized, and whether Inf or Nan values are produced.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`quaternion1`</b>: A tensor of shape `[A1, ... , An, 4]` storing normalized
  quaternions in its last dimension.
* <b>`quaternion2`</b>: A tensor of shape `[A1, ... , An, 4]` storing normalized
  quaternions in its last dimension.
* <b>`percent`</b>: A `float` or a tensor with a shape broadcastable to the shape `[A1,
  ... , An]`.
* <b>`eps`</b>: A `float` used to make operations safe. When left as None, the function
  automatically picks the best epsilon based on the dtype and the operation.
* <b>`name`</b>: A name for this op. Defaults to "quaternion_weights".


#### Raises:


* <b>`ValueError`</b>: If the shapes of quaternions do not match, if the last
  dimensions of quaternions are not 4, or if percent is neither a float, nor
  a tensor with last dimension 1.


#### Returns:

Two tensors of shape `[A1, ... , An, 1]` each, which are the two slerp
  weights for each quaternion.
