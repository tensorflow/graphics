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
"""Functions to extract tensors from proto files."""
import math
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.projects.points_to_3Dobjects.utils import tf_utils


# TODO(engelmann) instead use tf.io.decode_proto
decode_proto = tf.compat.v1.io.decode_proto


def centernet_proto_get(serialized):
  """Extracts the contents from a VoxelSample proto to tensors."""
  message_type = 'giotto_occluded_primitives.CenterNetSample'
  field_names = ['name', 'image_data', 'image_size', 'center2d', 'center3d',
                 'box_dims2d', 'box_dims3d', 'rotation', 'translation', 'rt',
                 'k']
  output_types = [tf.string, tf.string, tf.int32, tf.float32, tf.float32,
                  tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                  tf.float32]
  _, values = decode_proto(serialized, message_type, field_names, output_types)
  filename = tf.squeeze(values[0])
  image_data = tf.squeeze(values[1])
  image_size = tf.squeeze(values[2])
  center2d = tf.squeeze(values[3])
  center3d = tf.squeeze(values[4])
  box_dims2d = tf.squeeze(values[5])
  box_dims3d = tf.squeeze(values[6])
  rotation = tf.squeeze(values[7])
  translation = tf.squeeze(values[8])
  rt = tf.squeeze(values[9])
  k = tf.squeeze(values[10])
  image = tf.cast(tf.io.decode_image(image_data, channels=3), tf.float32)
  return filename, \
         image, \
         image_size, \
         center2d, \
         center3d, \
         box_dims2d, \
         box_dims3d, \
         rotation, \
         translation, \
         rt, \
         k


def decode_bytes(serialized):
  """Extracts the contents from a VoxelSample proto to tensors."""
  message_type = 'giotto_occluded_primitives.CenterNetSample'
  field_names = ['name', 'image_data', 'image_size', 'center2d', 'center3d',
                 'box_dims2d', 'box_dims3d', 'rotation', 'translation', 'rt',
                 'k']
  output_types = [tf.string, tf.string, tf.int32, tf.float32, tf.float32,
                  tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                  tf.float32]
  _, tensors = decode_proto(serialized, message_type, field_names, output_types)
  tensors = [tf.squeeze(t) for t in tensors]
  tensor_dict = dict(zip(field_names, tensors))
  tensor_dict['image'] = tf.cast(
      tf.io.decode_image(tensor_dict['image_data'], channels=3),
      tf.float32)
  tensor_dict['image'].set_shape([None, None, 3])
  tensor_dict['original_image_spatial_shape'] = \
    tf.shape(tensor_dict['image'])[:2]

  # Compute ground truth boxes. From (center2d,size2d) --> (ymin,xmin,ymax,xmax)
  box_min = tensor_dict['center2d'] - tensor_dict['box_dims2d']/2.0
  box_max = tensor_dict['center2d'] + tensor_dict['box_dims2d']/2.0
  box_min = tf.divide(box_min,
                      tf.cast(tensor_dict['original_image_spatial_shape'],
                              tf.float32))
  box_max = tf.divide(box_max,
                      tf.cast(tensor_dict['original_image_spatial_shape'],
                              tf.float32))
  tensor_dict['groundtruth_boxes'] = tf.reshape(
      tf.concat([box_min[1], box_min[0], box_max[1], box_max[0]], axis=0),
      [1, 4])

  tensor_dict['groundtruth_valid_classes'] = [1]
  tensor_dict['num_boxes'] = 1  # needs to be a scalar, no tensor
  tensor_dict['per_class_weight'] = [1]

  tensor_dict['box_dims3d'] = tf.reshape(tensor_dict['box_dims3d'], [1, 3])
  tensor_dict['size_offset3d'] = tensor_dict['box_dims3d'] - 0.5

  # Compute the dot on the ground plane used as prior for 3d pose (batched)
  image_size = tensor_dict['image_size']
  ray_2d = tf.cast(image_size, tf.float32) * [0.5, 3.0/4.0, 1/image_size[-1]]
  ray_2d = tf.reshape(ray_2d, [3, 1])
  k = tf.reshape(tensor_dict['k'], [3, 3])
  k_inv = tf.linalg.inv(k)
  rt = tf.reshape(tensor_dict['rt'], [3, 4])
  r = tf.gather(rt, [0, 1, 2], axis=1)
  t = tf.gather(rt, [3], axis=1)
  r_inv = tf.transpose(r, [1, 0])
  t_inv = tf.matmul(r_inv, t) * -1
  ray = tf.matmul(r_inv, tf.matmul(k_inv, ray_2d))
  l = -t_inv[-1] / ray[-1]  # determine lambda
  dot = tf.expand_dims(l, -1) * ray + t_inv
  tensor_dict['dot'] = tf.transpose(dot)
  tensor_dict['center3d'] = tf.reshape(tensor_dict['center3d'], [1, 3])
  tensor_dict['dot_offset3d'] = tensor_dict['center3d'] - tensor_dict['dot']

  # Rotation
  rad = tensor_dict['rotation'] * -1 * math.pi/180
  cos = tf.reshape(tf.cos(rad), [1, 1])
  sin = tf.reshape(tf.sin(rad), [1, 1])
  tensor_dict['rotation'] = tf.concat([cos - 1,
                                       sin - 0,
                                       -sin- 0,
                                       cos - 1], axis=1)
  return tensor_dict


def fix_rotation(rotation_y, object_class):
  if object_class == 2:  # table
    rotation_y = tf.math.floormod(rotation_y, math.pi)
  elif object_class == 3 or object_class == 4:  # bottle, bowl
    rotation_y = tf.constant(0.0)
  return rotation_y


def fix_rotation_matrix(rotation_matrix, object_class):
  fix_angle = tf_utils.euler_from_rotation_matrix(rotation_matrix, 1)
  if object_class == 2:  # table
    fix_angle = tf.math.floormod(fix_angle, math.pi)
  elif object_class == 3 or object_class == 4:  # bottle, bowl
    fix_angle = tf.constant(0.0)
  euler_angles_fixed = tf.stack([0.0, fix_angle, 0.0])
  rotation_matrix = rotation_matrix_3d.from_euler(euler_angles_fixed)
  return rotation_matrix


def decode_bytes_multiple(serialized):
  """Extracts the contents from a VoxelSample proto to tensors."""
  status = True
  if status:
    message_type = 'giotto_occluded_primitives.MultipleObjects'
    name_type_shape = {
        'name': [tf.string, []],
        'scene_filename': [tf.string, []],
        'image_data': [tf.string, []],
        'image_size': [tf.int32, []],
        'center2d': [tf.float32, [-1, 2]],
        'center3d': [tf.float32, [-1, 3]],
        'box_dims2d': [tf.float32, [-1, 2]],
        'box_dims3d': [tf.float32, [-1, 3]],
        'rotations_3d': [tf.float32, [-1, 3, 3]],
        'rt': [tf.float32, [3, 4]],
        'k': [tf.float32, []],
        'classes': [tf.int32, []],
        'mesh_names': [tf.string, []],
        'shapes': [tf.int32, []]
    }

    field_names = [n for n in name_type_shape.keys()]
    output_types = [name_type_shape[k][0] for k in name_type_shape.keys()]
    _, tensors = tf.io.decode_proto(serialized, message_type,
                                    field_names, output_types)

    # Unpack tensors into dict and reshape
    tensors = [tf.squeeze(t) for t in tensors]
    tensor_dict = dict(zip(name_type_shape.keys(), tensors))
    for name, [_, shape] in name_type_shape.items():
      if shape:
        tensor_dict[name] = tf.reshape(tensor_dict[name], shape)

    # Decode image and set shape
    image = tf.io.decode_image(tensor_dict['image_data'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([None, None, 3])
    original_image_spatial_shape = tf.shape(image)[:2]
    num_boxes = tf.shape(tensor_dict['center2d'])[0]

    # Compute ground truth boxes. From (center2d,size2d)-->(ymin,xmin,ymax,xmax)
    box_min = tensor_dict['center2d'] - tensor_dict['box_dims2d'] / 2.0
    box_max = tensor_dict['center2d'] + tensor_dict['box_dims2d'] / 2.0
    size = tf.cast(original_image_spatial_shape, tf.float32)
    size = tf.expand_dims(size, axis=0)
    size = tf.tile(size, [num_boxes, 1])
    box_min = tf.divide(box_min, size)
    box_max = tf.divide(box_max, size)
    groundtruth_boxes = tf.reshape(
        tf.concat([box_min[:, 1:2], box_min[:, 0:1],
                   box_max[:, 1:2], box_max[:, 0:1]], axis=1), [num_boxes, 4])

    # Compute the dot on the ground plane used as prior for 3D pose (batched)
    dot = tf.transpose(tf_utils.compute_dot(tensor_dict['image_size'],
                                            tensor_dict['k'],
                                            tensor_dict['rt']))

    dot_x = tf.transpose(tf_utils.compute_dot(tensor_dict['image_size'],
                                              tensor_dict['k'],
                                              tensor_dict['rt'],
                                              axis=1,
                                              image_intersection=(0.75, 0.75)))
    dot_x -= dot
    dot_x = tf.math.l2_normalize(dot_x)
    angle_y = tf.math.atan2(dot_x[0, 2], dot_x[0, 0])

    # [0: 'chair', 1: 'sofa', 2: 'table', 3: 'bottle', 4: 'bowl', 5: 'mug']):
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([3001627, 4256520, 4379243,
                              2876657, 2880940, 3797390]),
            values=tf.constant([0, 1, 2,
                                3, 4, 5])),
        default_value=tf.constant(-1),
        name='classes')
    classes = table.lookup(tensor_dict['classes'])

  rotations_3d = tensor_dict['rotations_3d']
  translations_3d = tensor_dict['center3d']
  sizes_3d = tensor_dict['box_dims3d']

  translation_correction = tf.concat(
      [tf.concat([tf.eye(3, dtype=tf.float32), tf.transpose(dot)], axis=-1),
       [[0.0, 0.0, 0.0, 1.0]]], axis=0)

  rt = tensor_dict['rt'] @ translation_correction

  # Transform to camera coordinate system
  rotation_left = rotation_matrix_3d.from_euler([0.0, angle_y, 0.0])
  rotation_right = rotation_matrix_3d.from_euler([0.0, -angle_y, 0.0])

  rotations_3d = rotation_left @ rotations_3d
  translations_3d = tf.transpose(
      rotation_left @ tf.transpose(translations_3d - dot))

  rotation_right = tf.concat([tf.concat(
      [rotation_right, [[0.0], [0.0], [0.0]]], axis=-1),
                              [[0.0, 0.0, 0.0, 1.0]]], axis=0)

  rt = rt @ rotation_right

  # Adapt rotation for table, bottle, bowl
  fix_rotation_matrix(rotations_3d[0], classes[0])
  rotations_3d = \
      tf.map_fn(fn=lambda t: fix_rotation_matrix(rotations_3d[t], classes[t]),
                elems=tf.range(num_boxes),
                fn_output_signature=tf.float32)
  rotations_3d = tf.reshape(rotations_3d, [-1, 9])

  status = True
  if status:
    output_dict = {}
    output_dict['name'] = tensor_dict['name']  # e.g. 'train-0000'
    output_dict['scene_filename'] = tensor_dict['scene_filename']
    output_dict['mesh_names'] = tensor_dict['mesh_names']
    output_dict['classes'] = tensor_dict['classes']
    output_dict['image'] = image
    output_dict['image_data'] = tensor_dict['image_data']
    output_dict['original_image_spatial_shape'] = original_image_spatial_shape
    output_dict['num_boxes'] = num_boxes
    output_dict['center2d'] = tensor_dict['center2d']
    output_dict['groundtruth_boxes'] = groundtruth_boxes
    output_dict['dot'] = dot - dot
    output_dict['sizes_3d'] = sizes_3d
    output_dict['translations_3d'] = translations_3d
    output_dict['rotations_3d'] = rotations_3d
    output_dict['rt'] = rt
    output_dict['k'] = tensor_dict['k']
    output_dict['groundtruth_valid_classes'] = classes
    output_dict['shapes'] = tensor_dict['shapes']

  return output_dict


def decode_bytes_multiple_scannet(serialized):
  """Extracts the contents from a VoxelSample proto to tensors."""
  status = True
  if status:
    message_type = 'giotto_occluded_primitives.MultipleObjects'
    name_type_shape = {
        'name': [tf.string, []],
        'scene_filename': [tf.string, []],
        'image_data': [tf.string, []],
        'image_size': [tf.int32, []],
        'center2d': [tf.float32, [-1, 2]],
        'center3d': [tf.float32, [-1, 3]],
        'box_dims2d': [tf.float32, [-1, 2]],
        'box_dims3d': [tf.float32, [-1, 3]],
        'rotations_3d': [tf.float32, [-1, 3, 3]],
        'rt': [tf.float32, [3, 4]],
        'k': [tf.float32, []],
        'classes': [tf.int32, []],
        'mesh_names': [tf.string, []],
        'shapes': [tf.int32, []]
    }

    field_names = [n for n in name_type_shape.keys()]
    output_types = [name_type_shape[k][0] for k in name_type_shape.keys()]
    _, tensors = tf.io.decode_proto(serialized, message_type,
                                    field_names, output_types)

    # Unpack tensors into dict and reshape
    tensors = [tf.squeeze(t) for t in tensors]
    tensor_dict = dict(zip(name_type_shape.keys(), tensors))
    for name, [_, shape] in name_type_shape.items():
      if shape:
        tensor_dict[name] = tf.reshape(tensor_dict[name], shape)

    # Decode image and set shape
    image = tf.io.decode_image(tensor_dict['image_data'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    width = tf.shape(image)[1]
    image_padded = tf.image.pad_to_bounding_box(image, 0, 0, width, width)
    image = image_padded
    image.set_shape([None, None, 3])
    original_image_spatial_shape = tf.stack(
        [tf.shape(image)[1], tf.shape(image)[0]], axis=0)
    tensor_dict['image_size'] = tf.stack([width, width, 3], axis=0)

    num_boxes = tf.shape(tensor_dict['center2d'])[0]

    # Compute ground truth boxes. (center2d,size2d) --> (ymin,xmin,ymax,xmax)
    box_min = tensor_dict['center2d'] - tensor_dict['box_dims2d'] / 2.0
    box_max = tensor_dict['center2d'] + tensor_dict['box_dims2d'] / 2.0
    size = tf.cast(original_image_spatial_shape, tf.float32)
    size = tf.expand_dims(size, axis=0)
    size = tf.tile(size, [num_boxes, 1])
    box_min = tf.divide(box_min, size)
    box_max = tf.divide(box_max, size)
    groundtruth_boxes = tf.reshape(
        tf.concat([box_min[:, 1:2], box_min[:, 0:1],
                   box_max[:, 1:2], box_max[:, 0:1]], axis=1), [num_boxes, 4])

    # Compute the dot on the ground plane used as prior for 3D pose (batched)
    dot = tf.transpose(tf_utils.compute_dot(tensor_dict['image_size'],
                                            tensor_dict['k'],
                                            tensor_dict['rt'],
                                            axis=1,
                                            image_intersection=(0.5, 0.6)))

    dot_x = tf.transpose(tf_utils.compute_dot(tensor_dict['image_size'],
                                              tensor_dict['k'],
                                              tensor_dict['rt'],
                                              axis=1,
                                              image_intersection=(0.6, 0.6)))
    dot_x -= dot
    dot_x = tf.math.l2_normalize(dot_x)
    angle_y = tf.math.atan2(dot_x[0, 2], dot_x[0, 0])

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([2818832, 2747177, 2871439, 2933112,
                              3001627, 3211117, 4256520, 4379243]),
            values=tf.constant([0, 1, 2, 3, 4, 5, 6, 7])),
        default_value=tf.constant(-1),
        name='classes')
    classes = table.lookup(tensor_dict['classes'])

  rotations_3d = tensor_dict['rotations_3d']  # (K, 3, 3)
  translations_3d = tensor_dict['center3d']
  sizes_3d = tensor_dict['box_dims3d']

  translation_correction = tf.concat(
      [tf.concat([tf.eye(3, dtype=tf.float32), tf.transpose(dot)], axis=-1),
       [[0.0, 0.0, 0.0, 1.0]]], axis=0)

  rt = tensor_dict['rt'] @ translation_correction

  # Transform to camera coordinate system
  rotation_left = rotation_matrix_3d.from_euler([0.0, angle_y, 0.0])
  rotation_right = rotation_matrix_3d.from_euler([0.0, -angle_y, 0.0])

  rotations_3d = rotation_left @ rotations_3d

  translations_3d = tf.transpose(
      rotation_left @ tf.transpose(translations_3d - dot))
  # translations_3d = translations_3d - dot

  rotation_right = tf.concat([tf.concat(
      [rotation_right, [[0.0], [0.0], [0.0]]], axis=-1),
                              [[0.0, 0.0, 0.0, 1.0]]], axis=0)

  rt = rt @ rotation_right

  # Adapt rotation for table, bottle, bowl
  # fix_rotation_matrix(rotations_3d[0], classes[0])
  num_boxes = tf.shape(tensor_dict['center2d'])[0]

  rotations_3d = tf.reshape(rotations_3d, [-1, 9])

  status = True
  if status:
    output_dict = {}
    output_dict['name'] = tensor_dict['name']  # e.g. 'train-0000'
    output_dict['scene_filename'] = tensor_dict['scene_filename']
    output_dict['mesh_names'] = tf.reshape(tensor_dict['mesh_names'], [-1])
    output_dict['classes'] = tf.reshape(tensor_dict['classes'], [-1])
    output_dict['image'] = image
    output_dict['image_data'] = tensor_dict['image_data']
    output_dict['original_image_spatial_shape'] = original_image_spatial_shape
    output_dict['num_boxes'] = num_boxes
    output_dict['center2d'] = tensor_dict['center2d']
    output_dict['groundtruth_boxes'] = groundtruth_boxes
    output_dict['dot'] = dot - dot
    output_dict['sizes_3d'] = sizes_3d
    output_dict['translations_3d'] = translations_3d
    output_dict['rotations_3d'] = rotations_3d
    output_dict['rt'] = rt
    output_dict['k'] = tensor_dict['k']
    output_dict['groundtruth_valid_classes'] = tf.reshape(classes, [-1])
    output_dict['shapes'] = tf.reshape(tensor_dict['shapes'], [-1])

  return output_dict
