# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the voxel prediction branch of Mesh R-CNN."""

from tensorflow.keras import layers as keras_layers

class MeshRefinementLayer(keras_layers.Layer):
  """Implements the 'Voxel Tube' from Mesh R-CNN."""

  def __init__(self,
               num_classes,
               num_convs,
               latent_dim,
               out_depth,
               name='VoxelPredictionLayer'):
    """Constructs the layer.

    Args:
      num_classes: `int` scalar denoting the number of different object classes
        that are to be predicted.
      num_convs: `int` scalar denoting the number of 2D convolutions to be used
        in the fully convolutional part of this layer.
      latent_dim: `int` scalar denoting the number of filters to be used for the
        2D convolutions.
      out_depth: `int` scalar denoting the depth of the predicted voxel
        occupancy probabilities.
      name: `String`, Optional name for the layer. Defaults to
        'VoxelPredictionLayer'.
    """
    super(MeshRefinementLayer, self).__init__(name=name)

    self.num_classes = num_classes
    self.num_convs = num_convs
    self.latent_dim = latent_dim
    self.out_depth = out_depth

    self.convs = []
    for c in range(self.num_convs):
      conv = keras_layers.Conv2D(self.latent_dim,
                                 3,
                                 activation='relu',
                                 name=f'{self.name}_FCN_Conv2D_{c}')
      self.convs.append(conv)

    self.deconv = keras_layers.Conv2DTranspose(self.latent_dim, 2, stride=2)
    # ToDo: output width and height??
    self.predictor = keras_layers.Conv2D(self.num_convs*self.out_depth, kernel_size=1)
