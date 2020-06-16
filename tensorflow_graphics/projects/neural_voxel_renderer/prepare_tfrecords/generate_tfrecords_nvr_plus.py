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
"""Script that generates the TFRecords for training/evaluating NVR plus."""
import os
import re
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer.prepare_tfrecords import data_pb2


decode_proto = tf.compat.v1.io.decode_proto


def voxel_proto_get(element):
  """Extracts the contents from a VoxelSample proto to tensors."""
  _, values = decode_proto(element,
                           'giotto_blender.VoxelSample',
                           ['name', 'voxel_data', 'size', 'obj_bbox'],
                           [tf.string, tf.string, tf.int32, tf.float32])
  filename = tf.squeeze(values[0])
  volume_raw = values[1]
  voxel_size = values[2]
  obj_bbox = values[3]

  voxels = tf.io.decode_raw(volume_raw, out_type=tf.uint8)
  voxels = tf.cast(tf.reshape(voxels, voxel_size), tf.float32) / 255.0
  return filename, voxels, voxel_size, obj_bbox


def blender_sample_proto_get(element):
  """Extracts the contents from a VoxelSample proto to tensors."""
  _, values = decode_proto(element,
                           'giotto_blender.BlenderSample',
                           ['name', 'image_data', 'height', 'width', 'obj',
                            'rotation', 'translation', 'obj_color',
                            'floor_color', 'light', 'envmap',
                            'elevation', 'rt', 'k'],
                           [tf.string, tf.string, tf.int32, tf.int32, tf.string,
                            tf.float32, tf.float32, tf.float32,
                            tf.float32, tf.float32, tf.string,
                            tf.float32, tf.float32, tf.float32])
  filename = tf.squeeze(values[0])
  image_data = tf.squeeze(values[1])
  obj_name = tf.squeeze(values[4])
  object_rotation = values[5]
  object_translation = tf.expand_dims(values[6], axis=0)
  floor_color = values[8]
  light_position = values[9]
  object_elevation = values[11]
  camera_extrinsics = tf.reshape(values[12], [3, 4])
  camera_intrinsics = tf.reshape(values[13], [3, 3])

  camera_rotation_matrix, camera_translation_vector = \
    tf.split(camera_extrinsics, [3, 1], axis=-1)
  return filename, image_data, obj_name, object_rotation, object_translation, floor_color, light_position, object_elevation, camera_rotation_matrix, camera_translation_vector, camera_intrinsics


def _expand_tfrecords_pattern(tfr_pattern):
  """Helper function to expand a tfrecord patter."""
  def format_shards(m):
    return '{}-?????-of-{:0>5}{}'.format(*m.groups())
  tfr_pattern = re.sub(r'^([^@]+)@(\d+)([^@]+)$', format_shards, tfr_pattern)
  return tfr_pattern


def tfrecords_to_dataset(tfrecords_pattern,
                         mapping_func,
                         batch_size,
                         buffer_size=5000):
  """Generates a TF Dataset from a rio pattern."""
  with tf.name_scope('Input/'):
    tfrecords_pattern = _expand_tfrecords_pattern(tfrecords_pattern)
    dataset = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=True)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(mapping_func)
    dataset = dataset.batch(batch_size)
    return dataset


VOXEL_SIZE = 128
IMAGE_SIZE = 256

BLENDER_SCALE = 2
DIAMETER = 4.2  # The voxel area in world coordinates
BACKGROUND_COLOR = 0.784
DATASET_SIZE = {'train': 40000, 'test': 1118}

flags.DEFINE_string('voxels_dir', '', 'Folder with the color voxel tfrecords.')
flags.DEFINE_string('images_dir', '', 'Folder with the images tfrecords.')
flags.DEFINE_string('output_dir', '', 'Folder to save the estimated tfrecords.')
flags.DEFINE_string('mode', 'train', '')
FLAGS = flags.FLAGS


def main(_):
  # ============================================================================
  # Load all colored voxels and place them in a volume table
  # ============================================================================
  color_voxels_tfrecord_dir = FLAGS.voxels_dir
  mode = 'test'
  batch_size = 1
  tfrecord_pattern = os.path.join(color_voxels_tfrecord_dir,
                                  '{0}@100.tfrecord'.format(mode))
  mapping_function = voxel_proto_get
  dataset = tfrecords_to_dataset(tfrecord_pattern,
                                 mapping_function,
                                 batch_size,
                                 buffer_size=1)
  volume_table = {}
  for (filename, voxels, _, _) in dataset:
    key = str(filename[0].numpy(), 'utf-8')
    volume_table[key] = voxels.numpy()

  # ============================================================================
  # Load blender dataset
  # ============================================================================
  blender_tfrecord_dir = FLAGS.images_dir  #
  blender_tfrecord = os.path.join(blender_tfrecord_dir,
                                  'default_chairs_{0}.tfrecord'.format(mode))
  batch_size = 1
  mapping_function = blender_sample_proto_get
  blender_dataset = tfrecords_to_dataset(blender_tfrecord,
                                         mapping_function,
                                         batch_size,
                                         buffer_size=1)

  # The camera parameters are fixed for the dataset.
  focal = np.array([284.44446, 284.44446], dtype=np.float32)
  principal_point = np.array([128, 128.], dtype=np.float32)

  # --------------------------------------------------------------------------
  # Where to save the estimated tf records.
  # WARNING: This will take up to 350GB!
  n_tfrecords = 100
  writer = [None]*n_tfrecords
  for tfrecord_id in range(n_tfrecords):
    writer_name = '{0}-{1:05d}-of-00100.tfrecord'.format(mode, tfrecord_id)
    writer[tfrecord_id] =\
      tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir, writer_name))

  for sample_id, sample in blender_dataset.enumerate():
    filename_tf, image_data_tf, obj_name_tf, object_rotation_tf, \
    object_translation_tf, floor_color_tf, light_position_tf,\
    object_elevation_tf, camera_rotation_matrix_tf, \
    camera_translation_vector_tf, _ = sample

    filename_np = str(filename_tf.numpy()[0], 'utf-8')
    image_data_np = image_data_tf.numpy()[0]
    obj_name = str(obj_name_tf.numpy()[0], 'utf-8')
    object_voxels = volume_table[obj_name]
    camera_rotation_matrix = camera_rotation_matrix_tf.numpy()
    camera_translation_vector = camera_translation_vector_tf.numpy()
    light_position = light_position_tf.numpy()[0]
    object_rotation = object_rotation_tf.numpy()
    object_translation = object_translation_tf.numpy()
    object_elevation = object_elevation_tf.numpy()
    ground_color = floor_color_tf.numpy()/255.

    # --------------------------------------------------------------------------
    # Place voxels in the scene
    # --------------------------------------------------------------------------
    object_rotation_v = object_rotation
    object_translation_v = object_translation[:, 0, [1, 0, 2]]*BLENDER_SCALE
    object_elevation_v = object_elevation

    ground_occupancy = np.zeros((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1),
                                dtype=np.float32)
    ground_occupancy[-2, 1:-2, 1:-2, 0] = 1
    ground_voxel_color = np.ones((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 3),
                                 dtype=np.float32)*ground_color
    #  np.array(GROUND_COLOR, dtype=np.float32)
    ground_voxel_color = np.concatenate([ground_voxel_color, ground_occupancy],
                                        axis=-1)

    scene_voxels = object_voxels*(1-ground_occupancy) + \
                   ground_voxel_color*ground_occupancy

    euler_angles_x = \
      np.deg2rad(180-object_rotation_v)*np.array([1, 0, 0], dtype=np.float32)
    euler_angles_y = \
      np.deg2rad(90-object_elevation_v)*np.array([0, 1, 0], dtype=np.float32)
    translation_vector = (object_translation_v/(DIAMETER*0.5))

    interpolated_voxels = helpers.object_to_world(scene_voxels,
                                                  euler_angles_x,
                                                  euler_angles_y,
                                                  translation_vector)

    # --------------------------------------------------------------------------
    # Estimate the ground image
    # --------------------------------------------------------------------------
    ground_image, ground_alpha = \
      helpers.generate_ground_image(IMAGE_SIZE, IMAGE_SIZE,
                                    focal, principal_point,
                                    camera_rotation_matrix,
                                    camera_translation_vector[:, :, 0],
                                    ground_color)

    # --------------------------------------------------------------------------
    # Render the voxels in the image plane
    # --------------------------------------------------------------------------
    object_rotation_dvr = np.array(np.deg2rad(object_rotation),
                                   dtype=np.float32)
    object_translation_dvr = np.array(object_translation[..., [0, 2, 1]],
                                      dtype=np.float32)
    object_translation_dvr -= np.array([0, 0, helpers.OBJECT_BOTTOM],
                                       dtype=np.float32)

    rerendering = \
      helpers.render_voxels_from_blender_camera(object_voxels,
                                                object_rotation_dvr,
                                                object_translation_dvr,
                                                256,
                                                256,
                                                focal,
                                                principal_point,
                                                camera_rotation_matrix,
                                                camera_translation_vector,
                                                absorption_factor=1.0,
                                                cell_size=1.0,
                                                depth_min=3.0,
                                                depth_max=5.0,
                                                frustum_size=(128, 128, 128))
    rerendering_image, rerendering_alpha = tf.split(rerendering, [3, 1],
                                                    axis=-1)

    rerendering_image = tf.image.resize(rerendering_image, (256, 256))
    rerendering_alpha = tf.image.resize(rerendering_alpha, (256, 256),
                                        method='nearest')

    final_composite = BACKGROUND_COLOR*(1-rerendering_alpha)*(1-ground_alpha)+\
                      ground_image*(1-rerendering_alpha)*ground_alpha + \
                      rerendering_image*rerendering_alpha
    final_composite = tf.cast(tf.squeeze(final_composite)*255, tf.uint8)
    final_composite_bytes = tf.image.encode_jpeg(final_composite, quality=99)

    # --------------------------------------------------------------------------
    # Save functionality
    interpolated_voxels_np = \
      (interpolated_voxels.numpy()[0]*255).astype(np.uint8)
    final_composite_bytes_np = final_composite_bytes.numpy()

    neural_voxel_sample = data_pb2.NeuralVoxelPlusSample()
    neural_voxel_sample.name = filename_np
    neural_voxel_sample.voxel_data.extend([interpolated_voxels_np.tobytes()])
    neural_voxel_sample.rerendering_data.extend([final_composite_bytes_np])
    neural_voxel_sample.image_data.extend([image_data_np])
    neural_voxel_sample.light_position.extend(list(light_position))
    tfrecord_id = sample_id//np.ceil(DATASET_SIZE[mode]/n_tfrecords)
    writer[tfrecord_id].write(neural_voxel_sample.SerializeToString())

  for tfrecord_id in range(n_tfrecords):
    writer[tfrecord_id].close()


if __name__ == '__main__':
  app.run(main)
