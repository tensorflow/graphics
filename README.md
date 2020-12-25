# TensorFlow Graphics

[![Build](https://github.com/tensorflow/graphics/workflows/Build/badge.svg?branch=master)](https://github.com/tensorflow/graphics/actions)
[![Code coverage](https://img.shields.io/coveralls/github/tensorflow/graphics.svg)](https://coveralls.io/github/tensorflow/graphics)
[![PyPI project status](https://img.shields.io/pypi/status/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
[![Supported Python version](https://img.shields.io/pypi/pyversions/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
[![PyPI release version](https://img.shields.io/pypi/v/tensorflow-graphics.svg)](https://pypi.org/project/tensorflow-graphics/)
[![Downloads](https://pepy.tech/badge/tensorflow-graphics)](https://pepy.tech/project/tensorflow-graphics)

The last few years have seen a rise in novel differentiable graphics layers
which can be inserted in neural network architectures. From spatial transformers
to differentiable graphics renderers, these new layers leverage the knowledge
acquired over years of computer vision and graphics research to build new and
more efficient network architectures. Explicitly modeling geometric priors and
constraints into neural networks opens up the door to architectures that can be
trained robustly, efficiently, and more importantly, in a self-supervised
fashion.

## Overview

At a high level, a computer graphics pipeline requires a representation of 3D
objects and their absolute positioning in the scene, a description of the
material they are made of, lights and a camera. This scene description is then
interpreted by a renderer to generate a synthetic rendering.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/graphics.jpg" width="600">
</div>

In comparison, a computer vision system would start from an image and try to
infer the parameters of the scene. This allows the prediction of which objects
are in the scene, what materials they are made of, and their three-dimensional
position and orientation.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv.jpg" width="600">
</div>

Training machine learning systems capable of solving these complex 3D vision
tasks most often requires large quantities of data. As labelling data is a
costly and complex process, it is important to have mechanisms to design machine
learning models that can comprehend the three dimensional world while being
trained without much supervision. Combining computer vision and computer
graphics techniques provides a unique opportunity to leverage the vast amounts
of readily available unlabelled data. As illustrated in the image below, this
can, for instance, be achieved using analysis by synthesis where the vision
system extracts the scene parameters and the graphics system renders back an
image based on them. If the rendering matches the original image, the vision
system has accurately extracted the scene parameters. In this setup, computer
vision and computer graphics go hand in hand, forming a single machine learning
system similar to an autoencoder, which can be trained in a self-supervised
manner.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv_graphics.jpg" width="600">
</div>

Tensorflow Graphics is being developed to help tackle these types of challenges
and to do so, it provides a set of differentiable graphics and geometry layers
(e.g. cameras, reflectance models, spatial transformations, mesh convolutions)
and 3D viewer functionalities (e.g. 3D TensorBoard) that can be used to train
and debug your machine learning models of choice.

## Installing TensorFlow Graphics

See the [install](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/install.md)
documentation for instructions on how to install TensorFlow Graphics.

## API Documentation

You can find the API documentation
[here](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/api_docs/python/tfg.md).

## Compatibility

TensorFlow Graphics is fully compatible with the latest stable release of
TensorFlow, tf-nightly, and tf-nightly-2.0-preview. All the functions are
compatible with graph and eager execution.

## Debugging

Tensorflow Graphics heavily relies on L2 normalized tensors, as well as having
the inputs to specific function be in a pre-defined range. Checking for all of
this takes cycles, and hence is not activated by default. It is recommended to
turn these checks on during a couple epochs of training to make sure that
everything behaves as expected. This
[page](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/debug_mode.md)
provides the instructions to enable these checks.

## Colab tutorials

To help you get started with some of the functionalities provided by TF
Graphics, some Colab notebooks are available below and roughly ordered by
difficulty. These Colabs touch upon a large range of topics including, object
pose estimation, interpolation, object materials, lighting, non-rigid surface
deformation, spherical harmonics, and mesh convolutions.

NOTE: the tutorials are maintained carefully. However, they are not considered
part of the API and they can change at any time without warning. It is not
advised to write code that takes dependency on them.

### Beginner

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb">Object pose estimation</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb">Camera intrinsics optimization</a></th>
    </tr>
    <tr>
      <td align="center">
        <a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb"><img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/notebooks/6dof_pose/thumbnail.jpg" width="200" height="200">
        </a>
      </td>
      <td align="center">
              <a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/intrinsics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/intrinsics/intrinsics_thumbnail.png" width="200" height="200">
        </a>
      </td>
    </tr>
  </table>
</div>

### Intermediate

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb">B-spline and slerp interpolation</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb">Reflectance</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb">Non-rigid surface deformation</a></th>
    </tr>
    <tr>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/interpolation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/interpolation/thumbnail.png" width="200" height="200"> </td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/reflectance.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/reflectance/thumbnail.png" width="200" height="200"></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/non_rigid_deformation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/non_rigid_deformation/thumbnail.jpg" width="200" height="200">
      </a></td>
    </tr>
  </table>
</div>

### Advanced

<div align="center">
  <table>
    <tr>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb">Spherical harmonics rendering</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb">Environment map optimization</a></th>
      <th style="text-align:center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_segmentation_demo.ipynb">Semantic mesh segmentation</a></th>
    </tr>
    <tr>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_approximation.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/sh_rendering/thumbnail.png" width="200" height="200">
      </a></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/spherical_harmonics_optimization.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/environment_lighting/thumbnail.png" width="200" height="200">
      </a></td>
      <td align="center"><a href="https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_segmentation_demo.ipynb"><img border="0" src="https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/thumbnail.jpg" width="200" height="200">
      </a></td>
    </tr>
  </table>
</div>

## TensorBoard 3D

Visual debugging is a great way to assess whether an experiment is going in the
right direction. To this end, TensorFlow Graphics comes with a TensorBoard
plugin to interactively visualize 3D meshes and point clouds.
[This demo](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/tensorboard/plugins/mesh/Mesh_Plugin_Tensorboard.ipynb)
shows how to use the plugin. Follow
[these instructions](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/tensorboard.md)
to install and configure TensorBoard 3D. Note that TensorBoard 3D is currently
not compatible with eager execution nor TensorFlow 2.

<div align="center">
  <img border="0"  src="https://storage.googleapis.com/tensorflow-graphics/git/readme/tensorboard_plugin.jpg" width="1280">
</div>

## Coming next...

Among many things, we are hoping to release resamplers, additional 3D
convolution and pooling operators, and a differentiable rasterizer!

Follow us on [Twitter](https://twitter.com/_TFGraphics_) to hear about the
latest updates!

## Additional Information

You may use this software under the
[Apache 2.0 License](https://github.com/tensorflow/graphics/blob/master/LICENSE).

## Community

As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

*   [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow): Ask
    or answer technical questions.
*   [GitHub](https://github.com/tensorflow/graphics/issues): Report bugs or make
    feature requests.
*   [TensorFlow Blog](https://blog.tensorflow.org/): Stay up to date on content
    from the TensorFlow team and best articles from the community.
*   [Youtube Channel](http://youtube.com/tensorflow/): Follow TensorFlow shows.

## References

If you use TensorFlow Graphics in your research, please reference it as:

    @inproceedings{TensorflowGraphicsIO2019,
       author = {Valentin, Julien and Keskin, Cem and Pidlypenskyi, Pavel and Makadia, Ameesh and Sud, Avneesh and Bouaziz, Sofien},
       title = {TensorFlow Graphics: Computer Graphics Meets Deep Learning},
       year = {2019}
    }

### Contact

Want to reach out? E-mail us at tf-graphics-contact@google.com!

### Contributors - in alphabetical order

-   Sofien Bouaziz (sofien@google.com)
-   Jay Busch
-   Forrester Cole
-   Ambrus Csaszar
-   Boyang Deng
-   Ariel Gordon
-   Christian Häne
-   Cem Keskin
-   Ameesh Makadia
-   Rohit Pandey
-   Romain Prévost
-   Pavel Pidlypenskyi
-   Stefan Popov
-   Konstantinos Rematas
-   Omar Sanseviero
-   Aviv Segal
-   Avneesh Sud
-   Andrea Tagliasacchi
-   Anastasia Tkach
-   Julien Valentin
-   He Wang
-   Yinda Zhang
