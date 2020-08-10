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
"""Mesh refinement branch of Mesh R-CNN"""

import tensorflow as tf
from tensorflow.keras import layers as keras_layers

from tensorflow_graphics.nn.layer.graph_convolution import \
  FeatureSteeredConvolutionKerasLayer
from tensorflow_graphics.projects.mesh_rcnn.ops.vertex_alignment import \
  vert_align


class MeshRefinementLayer(keras_layers.Layer):
  """
  Implements the Mesh refinement head of Mesh R-CNN.
  """

  def __init__(self,
               image_features_dim,
               vertex_feature_dim,
               latent_dim,
               stage_depth,
               initializer="normal"):
    """
    Args:
      image_features_dim: Dimension of features from vertex alignment.
      vertex_feature_dim: Dimension of previous stage. Can be 0.
      latent_dim: Output dimension for graph convolution layers.
      stage_depth: Integer, denoting number of graph convolutions to use.
      initializer: String, denoting how graph convolutions should be initialized.
    """
    super(MeshRefinementLayer, self).__init__()

    self.image_features_dim = image_features_dim
    self.vertex_feature_dim = vertex_feature_dim
    self.latent_dim = latent_dim
    self.stage_depth = stage_depth
    self.initializer = initializer

    self.bottleneck = keras_layers.Dense(self.latent_dim)
    self.refinement = keras_layers.Dense(3, activation='tanh')

  def call(self, inputs, **kwargs):
    pass


class MeshRefinementStage(keras_layers.Layer):
  """
  Implements one stage of the mesh refinement head of Mesh R-CNN.
  """

  def __init__(self,
               image_features_dim,
               vertex_feature_dim,
               latent_dim,
               stage_depth,
               initializer="normal"):
    """
    Args:
      image_features_dim: Dimension of features from vertex alignment.
      vertex_feature_dim: Dimension of previous stage. Can be 0.
      latent_dim: Output dimension for graph convolution layers.
      stage_depth: Integer, denoting number of graph convolutions to use.
      initializer: An initializer for the trainable variables. If `None`,
        defaults to `tf.compat.v1.truncated_normal_initializer(stddev=0.1)`.
    """
    super(MeshRefinementStage, self).__init__()

    self.image_features_dim = image_features_dim
    self.vertex_feature_dim = vertex_feature_dim
    self.latent_dim = latent_dim
    self.stage_depth = stage_depth
    self.initializer = initializer

    self.bottleneck = keras_layers.Dense(self.latent_dim, activation='relu')
    self.refinement = keras_layers.Dense(3, activation='tanh')

    self.gconvs = []
    for stage in range(self.stage_depth):
      if stage == 0:
        input_dim = self.latent_dim + self.vertex_feature_dim + 3
      else:
        input_dim = self.hidden_dim + 3

      gconv = FeatureSteeredConvolutionKerasLayer(translation_invariant=True,
                                                  input_dim=input_dim,
                                                  num_output_channels=self.hidden_dim,
                                                  initializer=self.initializer)
      self.gconvs.append(gconv)

  def call(self, inputs, **kwargs):
    feature = inputs['feature']
    mesh = inputs['mesh']
    intrinsics = inputs['intrinsics']
    vertex_features = inputs['vertex_features']
    aligned_features = vert_align(feature, mesh.get_padded(), intrinsics)
    flat_features = tf.reshape(aligned_features,
                               (-1, aligned_features.shape[-1]))
    image_features = self.bottleneck(flat_features)

    outputs = inputs
    if vertex_features is None:
      vertex_features = tf.concat([image_features, mesh.get_flattened()], 1)
    else:
      vertex_features = tf.concat(
          [vertex_features, image_features, mesh.get_flattened()], 1)

    for gconv in self.gconvs:
      new_vertex_features = gconv(vertex_features)
      outputs['vertex_features'] = new_vertex_features
      vertex_features = tf.concat([new_vertex_features, mesh.get_flattened()],
                                  1)

    offsets = self.refinement(vertex_features)
    outputs['mesh'] = mesh.add_offsets(offsets)

    return outputs
