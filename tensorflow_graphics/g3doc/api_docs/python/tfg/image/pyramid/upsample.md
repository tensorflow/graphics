<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.pyramid.upsample" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.pyramid.upsample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/pyramid.py">View source</a>



Generates the different levels of the pyramid (upsampling).

```python
tfg.image.pyramid.upsample(
    image, num_levels, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`image`</b>: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
  the height of the image, `W` the width of the image, and `C` the number of
  channels of the image.
* <b>`num_levels`</b>: The number of levels to generate.
* <b>`name`</b>: A name for this op that defaults to "pyramid_upsample".


#### Returns:

A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
`H_i` and `W_i` are the height and width of the upsampled image for the
level i.



#### Raises:


* <b>`ValueError`</b>: If the shape of `image` is not supported.