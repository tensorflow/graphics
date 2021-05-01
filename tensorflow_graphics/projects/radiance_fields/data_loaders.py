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
"""Data loader functions for the different datasets."""
import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf

DATASETDIR = '/data/synthetic/'


def load_nerf_dataset(dataset_dir=DATASETDIR,
                      dataset_name='lego',
                      split='train',
                      scale=1.0,
                      batch_size=10,
                      shuffle=True):
  """Load the synthetic data used in the NeRF paper."""
  anno_file = os.path.join(dataset_dir,
                           dataset_name,
                           'transforms_{}.json'.format(split))
  with tf.io.gfile.GFile(anno_file, 'r') as fp:
    anno = json.load(fp)

  init_width = 800
  height, width = int(800*scale), int(800*scale)

  images = np.zeros((len(anno['frames']), height, width, 4), dtype=np.float32)
  focals = np.zeros((len(anno['frames']), 2), dtype=np.float32)
  principal_points = np.zeros((len(anno['frames']), 2), dtype=np.float32)
  transform_matrices = np.zeros((len(anno['frames']), 4, 4), dtype=np.float32)

  for i, frame in enumerate(anno['frames']):
    image_name = os.path.join(dataset_dir,
                              dataset_name,
                              frame['file_path'] + '.png')
    with tf.io.gfile.GFile(image_name, 'rb') as f:
      im = Image.open(f)
      if scale != 1.0:
        im = im.resize((width, height))
      im = (np.array(im)/255.).astype(np.float32)
      images[i, :, :, :] = im

    camera_angle_x = float(anno['camera_angle_x'])
    focal = (.5 * init_width / np.tan(.5 * camera_angle_x))*scale
    focals[i, :] = [focal, focal]
    principal_points[i, :] = [width/2., height/2.]

    transform_matrix = np.array(frame['transform_matrix'])
    transform_matrices[i, :, :] = transform_matrix

  dataset = tf.data.Dataset.from_tensor_slices((images,
                                                focals,
                                                principal_points,
                                                transform_matrices))
  dataset = dataset.batch(batch_size)
  if shuffle:
    dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

  return dataset, height, width


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_srn_sample(path_to_srn, category, split, shape_name, view_id):
  """Load image and camera parameters from the SRN-ShapeNet dataset."""
  image_name = os.path.join(path_to_srn,
                            '{0}_{1}'.format(category, split),
                            shape_name,
                            'rgb',
                            '{0:06d}.png'.format(view_id))
  with tf.io.gfile.GFile(image_name, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)/255.
  if image.shape[-1] == 3:
    image = np.dstack([image, np.ones((128, 128, 1), dtype=np.float32)])
  color_sum = np.sum(image[..., :3], axis=-1)
  i, j = (color_sum == 3).nonzero()
  image[i, j, -1] = 0

  intrinsics_fname = os.path.join(path_to_srn,
                                  '{0}_{1}'.format(category, split),
                                  shape_name,
                                  'intrinsics.txt')
  with tf.io.gfile.GFile(intrinsics_fname, 'r') as f:
    line = f.readline().replace('\n', '').split(' ')
    intrinsics_info = np.array(line, dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = intrinsics_info[0]
    intrinsics[1, 1] = intrinsics_info[0]
    intrinsics[0, 2] = intrinsics_info[1]
    intrinsics[1, 2] = intrinsics_info[2]

  extrinsics_fname = os.path.join(path_to_srn,
                                  '{0}_{1}'.format(category, split),
                                  shape_name,
                                  'pose',
                                  '{0:06d}.txt'.format(view_id))
  with tf.io.gfile.GFile(extrinsics_fname, 'r') as f:
    extrinsics = np.loadtxt(f, dtype=float).reshape([4, 4]).astype(np.float32)

  extrinsics = np.linalg.inv(extrinsics)
  rotation_matrix = extrinsics[:3, :3]
  translation_vector = extrinsics[:3, [3]]

  focal = intrinsics[[0, 1], [0, 1]]
  principal_point = intrinsics[[0, 1], [2, 2]]
  return image, focal, principal_point, rotation_matrix, translation_vector


def set_srn_sample_proto(shape_name, shape_id, view,
                         image, focal, principal_point,
                         rotation_matrix, translation_vector,
                         w2v_alpha, w2v_beta):
  """Set a proto message (tf.Example) containing rendering information from SRN-ShapeNet.

  Args:
      shape_name (str): Name of the 3D shape that the rendering came from.
      shape_id (int): Index of the 3D shape.
      view (int): Index of the camera view.
      image (float): A numpy array of size [height, witdth, 3].
      focal (float): A numpy array of size [2].
      principal_point (float): A numpy array of size [2].
      rotation_matrix (float): A numpy array of size [3, 3].
      translation_vector (float): A numpy array of size [3, 1].
      w2v_alpha (float): A numpy array of size [1, 3].
      w2v_beta (float): A numpy array of size [1, 3].

  Returns:
      A tf.Example proto message.
  """
  image_bytes = tf.io.encode_png((image*255).astype(np.uint8))

  feature = {
      'name': _bytes_feature(shape_name.encode('utf_8')),
      'shape_index': _int64_feature(shape_id),
      'view': _int64_feature(view),
      'image_data': _bytes_feature(image_bytes),
      'image_size': _int64_feature(list(image.shape)),
      'focal': _float_feature(list(focal)),
      'principal_point': _float_feature(list(principal_point)),
      'rotation_matrix': _float_feature(list(rotation_matrix.flatten())),
      'translation_vector': _float_feature(list(translation_vector.flatten())),
      'w2v_alpha': _float_feature(list(w2v_alpha.flatten())),
      'w2v_beta': _float_feature(list(w2v_beta.flatten()))
      }
  proto_sample = tf.train.Example(features=tf.train.Features(feature=feature))
  return proto_sample


def get_srn_sample_proto(element):
  """Converts a tf.Example to a list of tensors."""
  feature_description = {
      'name': tf.io.FixedLenFeature([], tf.string),
      'view': tf.io.FixedLenFeature([], tf.int64),
      'shape_index': tf.io.FixedLenFeature([], tf.int64),
      'image_data': tf.io.FixedLenFeature([], tf.string),
      'image_size': tf.io.FixedLenFeature([3], tf.int64),
      'focal': tf.io.FixedLenFeature([2], tf.float32),
      'principal_point': tf.io.FixedLenFeature([2], tf.float32),
      'rotation_matrix': tf.io.FixedLenFeature([9], tf.float32),
      'translation_vector': tf.io.FixedLenFeature([3], tf.float32),
      'w2v_alpha': tf.io.FixedLenFeature([3], tf.float32),
      'w2v_beta': tf.io.FixedLenFeature([3], tf.float32),
  }

  values = tf.io.parse_single_example(element, feature_description)

  filename = tf.squeeze(values['name'])
  view = tf.squeeze(values['view'])
  shape_index = tf.squeeze(values['shape_index'])
  image_data = tf.squeeze(values['image_data'])
  # image_size = tf.squeeze(values[4])
  focal = tf.squeeze(values['focal'])
  principal_point = tf.squeeze(values['principal_point'])
  rotation_matrix = tf.squeeze(values['rotation_matrix'])
  translation_vector = tf.squeeze(values['translation_vector'])
  w2v_alpha = tf.squeeze(values['w2v_alpha'])
  w2v_beta = tf.squeeze(values['w2v_beta'])

  image = tf.cast(tf.io.decode_png(image_data), tf.float32)/255.
  rotation_matrix = tf.reshape(rotation_matrix, [3, 3])
  translation_vector = tf.reshape(translation_vector, [3, 1])

  return filename, view, shape_index, image, focal, principal_point, rotation_matrix, translation_vector, w2v_alpha, w2v_beta
