<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.bspline" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.math.interpolation.bspline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/bspline.py">View source</a>



Tensorflow.graphics B-spline interpolation module.


This module supports cardinal B-spline interpolation up to degree 4, with up
to C3 smoothness. It has functions to calculate basis functions, control point
weights, and the final interpolation.

## Classes

[`class Degree`](../../../tfg/math/interpolation/bspline/Degree.md): Defines valid degrees for B-spline interpolation.

## Functions

[`interpolate(...)`](../../../tfg/math/interpolation/bspline/interpolate.md): Applies B-spline interpolation to input control points (knots).

[`interpolate_with_weights(...)`](../../../tfg/math/interpolation/bspline/interpolate_with_weights.md): Interpolates knots using knot weights.

[`knot_weights(...)`](../../../tfg/math/interpolation/bspline/knot_weights.md): Function that converts cardinal B-spline positions to knot weights.

