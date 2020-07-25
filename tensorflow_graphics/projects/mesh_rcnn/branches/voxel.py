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

import tensorflow as tf

def voxel_loss(predicted_voxel_logits, instances, loss_weight=1.):
  """
  Computes the voxel loss defined in the Mesh R-CNN paper
  (https://arxiv.org/pdf/1906.02739.pdf).

  Args:
    predicted_voxel_logits: Tensor of shape (B, C, D, H , W), where B is the
    total number of predicted voxels in all images, C is the number of
    foreground classes, and D, H, W are depth, height and width dimensions
    of the predictions. The values are logits.
    instances: A list of N instances, where N denotes the number of images in
    the batch. There must be an instance for each prediction.
    Ground truth labels (class, box, mask, ...) are stored in fields.
      ToDo: Be more precise regarding structure of instances.
    loss_weight: Float that is multiplied with the actual loss value.
    Defaults to 1.0.

  Returns:
    Scalar tensor containing the weighted loss.
  """
  is_class_agnostic = predicted_voxel_logits.shape[1] == 1
  number_of_voxels = predicted_voxel_logits.shape[0]
  voxel_size = predicted_voxel_logits.shape[2]

  if predicted_voxel_logits.shape[3] != voxel_size or predicted_voxel_logits.shape[4] != voxel_size:
    raise ValueError("Voxel predictions must be square!")
