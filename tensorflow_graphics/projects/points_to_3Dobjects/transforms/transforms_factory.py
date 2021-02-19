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
"""Transforms factory."""
import functools

from tensorflow_graphics.projects.points_to_3Dobjects.transforms import preprocessor
from tensorflow_graphics.projects.points_to_3Dobjects.transforms import targets
from tensorflow_graphics.projects.points_to_3Dobjects.transforms import transforms


class TransformsFactory:
  """Transrofms factory class."""

  def __int__(self):
    pass

  @staticmethod
  def get_transform_group(name, params):
    """Get transform."""
    input_image = 'image'
    original_image_shape = 'original_image_spatial_shape'
    groundtruth_boxes = 'groundtruth_boxes'
    groundtruth_instance_masks = 'groundtruth_instance_masks'
    num_boxes = 'num_boxes'
    valid_classes = 'groundtruth_valid_classes'

    if name == 'affine_transform':
      if 'random' not in params:
        params['random'] = False
      preprocess_options = [(transforms.affine_transform, {
          'image_size': params['image_size'],
          'transform_gt_annotations': params['transform_gt_annotations'],
          'random': params['random'],
          'random_side_scale_range': (0.6, 1.4, 0.1),
          'random_flip_probability': 0.5
      })]
      func_arg_map = {
          transforms.affine_transform: (
              (input_image, original_image_shape, groundtruth_boxes,
               groundtruth_instance_masks), (input_image, original_image_shape,
                                             groundtruth_boxes,
                                             groundtruth_instance_masks))
      }
    elif name == 'centernet_preprocessing':
      if 'random' not in params:
        params['random'] = False
      preprocess_options = [
          (transforms.rgb_to_bgr, {}),
          (transforms.affine_transform, {
              'image_size': params['image_size'],
              'transform_gt_annotations': params['transform_gt_annotations'],
              'random': params['random'],
              'random_side_scale_range': (0.6, 1.4, 0.1),
              'random_flip_probability': 0.5
          }),
          (transforms.subtract_mean_and_normalize, {
              'means': [0.40789655, 0.44719303, 0.47026116],
              'std': [0.2886383, 0.27408165, 0.27809834],
              'random': params['random']
          })
      ]
      func_arg_map = {
          transforms.rgb_to_bgr: (input_image,),
          transforms.affine_transform: (
              (input_image, original_image_shape, groundtruth_boxes,
               groundtruth_instance_masks), (input_image, original_image_shape,
                                             groundtruth_boxes,
                                             groundtruth_instance_masks)),
          transforms.subtract_mean_and_normalize: (input_image,),
      }
    elif name == 'centernet_train_targets':
      preprocess_options = [
          (targets.assign_center_targets, {
              'image_size': params['image_size'],
              'stride': params['stride'],
              'num_classes': params['num_classes']
          }),
          (targets.assign_offset_targets, {
              'image_size': params['image_size'],
              'stride': params['stride'],
          }),
          (targets.assign_width_height_targets, {
              'image_size': params['image_size'],
              'stride': params['stride'],
          }),
          (targets.assign_valid_boxes_mask_targets, {
              'image_size': params['image_size'],
              'stride': params['stride'],
          }),
          (targets.assign_center_indices_targets, {
              'image_size': params['image_size'],
              'stride': params['stride'],
          }),
      ]
      func_arg_map = {
          targets.assign_center_targets: (
              (groundtruth_boxes, valid_classes, num_boxes),
              ('centers',)),
          targets.assign_offset_targets: (
              (groundtruth_boxes,
               num_boxes), ('offset',)),
          targets.assign_width_height_targets: (
              (groundtruth_boxes, num_boxes),
              ('width_height',)),
          targets.assign_valid_boxes_mask_targets: (
              (groundtruth_boxes, num_boxes),
              ('valid_boxes_mask',)),
          targets.assign_center_indices_targets: (
              (groundtruth_boxes,
               num_boxes), ('indices',))
      }
    elif name == 'classification_preprocessing':
      return None
    elif name == 'classification_targets':
      return None
    else:
      raise ValueError(f'Transform not available {name}')

    transform_fn = functools.partial(
        preprocessor.preprocess,
        preprocess_options=preprocess_options,
        func_arg_map=func_arg_map)

    return transform_fn
