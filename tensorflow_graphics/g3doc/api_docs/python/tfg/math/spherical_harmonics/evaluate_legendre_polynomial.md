<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.evaluate_legendre_polynomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.evaluate_legendre_polynomial

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Evaluates the Legendre polynomial of degree l and order m at x.

```python
tfg.math.spherical_harmonics.evaluate_legendre_polynomial(
    degree_l, order_m, x
)
```



<!-- Placeholder for "Used in" -->


#### Note:

This function is implementing the algorithm described in p. 10 of `Spherical
Harmonic Lighting: The Gritty Details`.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`degree_l`</b>: An integer tensor of shape `[A1, ..., An]` corresponding to the
  degree of the associated Legendre polynomial. Note that `degree_l` must be
  non-negative.
* <b>`order_m`</b>: An integer tensor of shape `[A1, ..., An]` corresponding to the
  order of the associated Legendre polynomial. Note that `order_m` must
  satisfy `0 <= order_m <= l`.
* <b>`x`</b>: A tensor of shape `[A1, ..., An]` with values in [-1,1].


#### Returns:

A tensor of shape `[A1, ..., An]` containing the evaluation of the legendre
polynomial.
