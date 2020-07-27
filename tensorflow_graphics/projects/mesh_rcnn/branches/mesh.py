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
from tensorflow_graphics.nn.layer import graph_convolution


class MeshRefinementLayer(tf.keras.layers.Layer):
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


  def call(self, inputs, **kwargs):
    pass
