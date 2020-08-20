## Neural Voxel Renderer: Learning and accurate and controllable rendering tool

[Konstantinos Rematas](http://www.krematas.com/) and [Vittorio Ferrari](https://sites.google.com/corp/view/vittoferrari)<br>
CVPR 2020<br>
[Paper](https://arxiv.org/abs/1912.04591)

### Introduction

This directory is based on our CVPR paper [Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool](https://arxiv.org/abs/1912.04591).

We present a neural rendering framework that maps a voxelized scene into a high
quality image. Highly-textured objects and scene element interactions are
realistically rendered by our method, despite having a rough representation as
an input. Moreover, our approach allows controllable rendering: geometric and
appearance modifications in the input are accurately propagated to the output.
The user can move, rotate and scale an object, change its appearance and texture
or modify the position of the light and all these edits are represented in the
final rendering. We demonstrate the effectiveness of our approach by rendering
scenes with varying appearance, from single color per object to complex,
high-frequency textures. We show that our rerendering network can generate very
detailed images that represent precisely the appearance of the input scene. Our
experiments illustrate that our approach achieves more accurate image synthesis
results compared to alternatives and can also handle low voxel grid resolutions.
Finally, we show how our neural rendering framework can capture and faithfully
render objects from real images and from a diverse set of classes.

### Getting started

The code has been tested with Python 3.6 and TensorFlow 2. See the
[inference demo](https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/projects/neural_voxel_renderer/demo.ipynb)
for running a forward pass of our network using pretrained weights, or
[training demo](https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/projects/neural_voxel_renderer/train.ipynb)
for running the training procedure. Note that the code has been modified
compared to the original paper version: here we use differentiable volumetric
rendering instead of splatting thanks to Romain Prevost; the code runs on
TensorFlow 2 but the original training and evaluation was done in TensorFlow
1.14; the non-visible voxels take the color of the nearest visible voxel.

### References
If you find our code or paper useful, please consider citing

    @inproceedings{RematasCVPR2020,
      title = {Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool},
      author = {Konstantinos Rematas and Vittorio Ferrari},
      booktitle = {CVPR},
      year = {2020}
    }

### Contact

Please contact [Konstantinos Rematas](mailto:krematas@gmail.com)
