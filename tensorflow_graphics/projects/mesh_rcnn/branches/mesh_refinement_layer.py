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
"""Mesh refinement layer of Mesh R-CNN"""

from tensorflow.keras import layers as keras_layers

from tensorflow_graphics.projects.mesh_rcnn.branches.mesh_refinement_stage import MeshRefinementStage


class MeshRefinementLayer(keras_layers.Layer):
  """Implements the Mesh refinement head of Mesh R-CNN."""

  def __init__(self,
               num_stages,
               num_gconvs,
               gconv_dim,
               gconv_init='normal',
               name='MeshRefinementHead',
               **kwargs):
    """
    Args:
      num_stages: Number of refinement stages.
      num_gconvs: Number of graph convolutions per stage.
      gconv_dim: output dimensionality of graph convolutions.
      gconv_init: An initializer for the graph convolutions
        (see keras.initializers).

      name: An optional name for the layer.
      **kwargs: Optional keyword arguments that get passed to the base layer.
    """
    super(MeshRefinementLayer, self).__init__(name=name, **kwargs)

    self.n_stages = num_stages
    self.n_gconvs = num_gconvs
    self.gconv_dim = gconv_dim
    self.gconv_init = gconv_init

    self.stages = []

  def build(self, input_shape):
    for i in range(self.n_stages):
      vertex_features_dim = 0 if i == 0 else self.gconv_dim
      stage = MeshRefinementStage(image_features_dim=input_shape[-1],
                                  vertex_feature_dim=vertex_features_dim,
                                  latent_dim=self.gconv_dim,
                                  stage_depth=self.n_gconvs,
                                  initializer=self.gconv_init,
                                  name=f'MeshRefinementStage_{i}')

      self.stages.append(stage)

  def call(self, features, mesh, intrinsics, **kwargs):
    """Forward pass of the layer.

    Args:
      features: image features from which to extract features in VertAlign
      mesh: `Meshes` object containing the meshes of the current batch.
      intrinsics: float32 tensor of shape `[N, 3, 3]` containing the
        intrinsic matrices.
      **kwargs: Optional keyword arguments.

    Returns:
      The refined `Meshes`.
    """
    if mesh.is_empty:
      return mesh

    inputs = {
        'feature': features,
        'mesh': mesh,
        'intrinsics': intrinsics,
        'vertex_features': None
    }

    for stage in self.stages:
      inputs = stage(inputs)

    return mesh
