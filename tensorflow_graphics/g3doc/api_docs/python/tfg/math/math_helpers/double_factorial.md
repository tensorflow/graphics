<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.math_helpers.double_factorial" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.math_helpers.double_factorial

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/math_helpers.py">View source</a>



Computes the double factorial of `n`.

```python
tfg.math.math_helpers.double_factorial(
    n
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`n`</b>: A tensor of shape `[A1, ..., An]` containing positive integer values.


#### Returns:

A tensor of shape `[A1, ..., An]` containing the double factorial of `n`.
