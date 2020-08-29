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
"""The Mesh R-CNN Architecture as Keras Model."""

import tensorflow as tf
from tensorflow import keras as K

from tensorflow_graphics.projects.mesh_rcnn.branches.mesh_refinement_layer import \
  MeshRefinementLayer
from tensorflow_graphics.projects.mesh_rcnn.branches.voxel_layer import \
  VoxelPredictionLayer
from tensorflow_graphics.projects.mesh_rcnn.loss import mesh_rcnn_loss
from tensorflow_graphics.projects.mesh_rcnn.ops.cubify import cubify


class MeshRCNN(K.Model):
  """Implementation of Mesh R-CNN as a Keras Model.

  Currently, this implementation contains only the 3D prediction part of Mesh
  R-CNN. As input, it expects convolutional outputs from some 2D backbone
  Network, RoIAligned with predicted bounding boxes. These features are used to
  predict voxel occupancy scores as a bridge between the 2D features and the
  rich 3D structure of the output meshes that this model computes.

  Citation:
    @InProceedings{Gkioxari_2019_ICCV,
      author = {Gkioxari, Georgia and Malik, Jitendra and Johnson, Justin},
      title = {Mesh R-CNN},
      booktitle = {Proceedings of the IEEE/CVF International Conference on
        Computer Vision (ICCV)},
      month = {October},
      year = {2019}
    }
  """

  def __init__(self, config, name='MeshRCNN'):
    super(MeshRCNN, self).__init__(name=name)

    self.cubify_threshold = config.cubify_threshold

    self.voxel_prediction = VoxelPredictionLayer(
        num_convs=config.num_convs_voxel_layer,
        latent_dim=config.latent_dim_voxel_layer,
        out_depth=config.voxel_grid_depth,
        name=config.voxel_layer_name
    )
    self.mesh_refinement = MeshRefinementLayer(
        num_stages=config.mesh_refinement_num_stages,
        num_gconvs=config.mesh_refinement_num_gconvs,
        gconv_dim=config.mesh_refinement_gconv_dim,
        gconv_init=config.mesh_refinement_gconv_inititalizer,
        name=config.mesh_refinement_layer_name
    )

  def call(self, inputs, training=None, mask=None):
    """Performs the forward pass of the model.

    Args:
      inputs: List of two tensors: First tensor contains RoIAligned
        convolutional features from some 2D backbone in the shape
        `[batch_size, height, width, num_features]`. Second tensor is a float32
        tensor of shape `[batch_size, 3, 3]` containing the intrinsic camera
        parameters for each input feature.
      training: Boolean or boolean scalar tensor, indicating whether to run
        the `Network` in training mode or inference mode.
      mask: A mask or list of masks. A mask can be
        either a tensor or None (no mask).


    Returns:
      The 3D outputs as a batch of predicted voxel grids and a batch of 3D
      Triangle Meshes.
    """
    # ToDo check inputs
    voxel_occupancy_probabilities = self.voxel_prediction(inputs[0])
    init_mesh = cubify(voxel_occupancy_probabilities, self.cubify_threshold)
    meshes = self.mesh_refinement(inputs[0], init_mesh, inputs[1])

    return voxel_occupancy_probabilities, meshes

  def compile(self,
              optimizer='adam',
              metrics=None,
              loss_weights=None,
              mesh_loss_sample_size_gt=5000,
              mesh_loss_sample_size_pred=5000,
              **kwargs):
    """Compiles the Mesh R-CNN Model and initializes all Loss functions, metrics
    and the optimizer.

    Args:
      optimizer: String (name of optimizer) or optimizer instance. See
        `tf.keras.optimizers`.
      metrics: List of metrics to be evaluated by the model during training
        and testing.
      loss_weights: Dictionary containing the loss weights for the different
        losses of Mesh R-CNN. Must map the following keys to float values:
        {
          'voxel': <`float32` scalar>,
          'chamfer': <`float32` scalar>,
          'normal': <`float32` scalar>,
          'edge': <`float32` scalar
        }
      mesh_loss_sample_size_gt: 'int' scalar denoting the number of points to be
        sampled from the mesh surfaces of the ground truth shapes for the
        evaluation of the mesh loss.
      mesh_loss_sample_size_pred: 'int' scalar denoting the number of points to be
        sampled from the mesh surfaces of the predicted shapes for the
        evaluation of the mesh loss.

    Raises:
      ValueError: In case of invalid arguments for
      `optimizer`, `metrics` or `loss_weights`.
    """
    super(MeshRCNN, self).compile(optimizer=optimizer, metrics=metrics)
    self.loss_weights = loss_weights
    self.voxel_loss = K.losses.BinaryCrossentropy()
    self.mesh_loss = mesh_rcnn_loss.initialize(
        self.loss_weights,
        gt_sample_size=mesh_loss_sample_size_gt,
        pred_sample_size=mesh_loss_sample_size_pred)

  def train_step(self, data):
    """Logic of one train step.

    Args:
      data: A nested structure of `Tensor`s containing the input features
        together with the intrinsic camera parameters and the ground truth
        voxel grid and ground truth meshes.

    Returns:
      A `dict` containing the computed losses in the form:
      `{'voxel_loss': 0.2, 'mesh_loss': 0.7}`.
    """
    inputs, ground_truths = data  # ToDo check format
    voxel_gt, mesh_gt = ground_truths

    with tf.GradientTape() as tape:
      # ToDo intermediary mesh predictions from single stages???
      voxel_predictions, mesh_predictions = self(inputs, training=True)
      voxel_loss = self.voxel_loss(voxel_gt, voxel_predictions)
      mesh_loss = self.mesh_loss(mesh_gt, mesh_predictions)

    voxel_gradients = tape.gradient(voxel_loss,
                                    self.voxel_prediction.trainable_weights)
    mesh_gradients = tape.gradient(mesh_loss,
                                   self.mesh_refinement.trainable_weights)

    self.optimizer.apply_gradients(zip(mesh_gradients,
                                       self.mesh_refinement.trainable_weights))
    self.optimizer.apply_gradients(zip(voxel_gradients,
                                       self.voxel_prediction.trainable_weights))

    return {'voxel_loss': voxel_loss, 'mesh_loss': mesh_loss}

  def test_step(self, data):
    """Logic of one test step.

    Args:
      data: A nested structure of `Tensor`s containing the input features
        together with the intrinsic camera parameters and the ground truth
        voxel grid and ground truth meshes.

    Returns:
      A `dict` containing the computed metrics in the form:
      `{'voxel_loss': 0.2, 'mesh_loss': 0.7}`.
    """
    inputs, ground_truths = data  # ToDo check format
    voxel_gt, mesh_gt = ground_truths

    voxel_predictions, mesh_predictions = self(inputs, training=False)

    voxel_loss = self.voxel_loss(voxel_gt, voxel_predictions)
    mesh_loss = self.mesh_loss(mesh_gt, mesh_predictions)

    self.compiled_metrics.update_state(inputs,
                                       (voxel_predictions, mesh_predictions))

    self.compiled_metrics['voxel_loss'] = voxel_loss
    self.compiled_metrics['mesh_loss'] = mesh_loss

    return {m.name: m.result() for m in self.metrics}
