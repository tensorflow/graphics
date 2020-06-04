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
"""Example run of the NVR+ model."""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.voxels import visual_hull

tf.disable_eager_execution()

BLENDER_SCALE = 2
DIAMETER = 4.2
PATH_TO_DATA = '/google/src/cloud/krematas/voxel_color/google3/third_party/py/tensorflow_graphics/projects/neural_voxel_renderer/data/example_data.p'


def main(_):
  with open(PATH_TO_DATA, 'rb') as f:
    example_data = pickle.load(f)

  object_voxels = example_data['voxels']
  camera_intrinsics = example_data['camera_intrinsics']
  camera_rotation_matrix = example_data['camera_rotation_matrix']
  camera_translation_vector = example_data['camera_translation_vector']

  # ============================================================================
  ground_color = np.array((136., 162, 199))/255.

  ground_image, ground_alpha = \
    helpers.generate_ground_image(camera_intrinsics,
                                  camera_rotation_matrix,
                                  camera_translation_vector,
                                  ground_color)

  with tf.Session() as sess:
    ground_image_np, ground_alpha_np = sess.run([ground_image, ground_alpha])

  plt.imshow(ground_image_np*ground_alpha_np)
  plt.show()

  # ============================================================================
  object_rotation_dvr = np.array([np.deg2rad(example_data['object_rotation'])],
                                 dtype=np.float32)
  object_translation_dvr = \
    np.array(example_data['object_translation'][:, [0, 2, 1]], dtype=np.float32)
  object_translation_dvr -= np.array([0, 0, helpers.OBJECT_BOTTOM],
                                     dtype=np.float32)

  focal = camera_intrinsics[[0, 1], [0, 1]]
  principal_point = camera_intrinsics[:2, 2]

  rerendering = \
    helpers.render_voxels_from_blender_camera(object_voxels,
                                              object_rotation_dvr,
                                              object_translation_dvr,
                                              focal,
                                              principal_point,
                                              camera_rotation_matrix,
                                              camera_translation_vector,
                                              absorption_factor=1.0,
                                              cell_size=1.0,
                                              depth_min=1.0,
                                              depth_max=5.0,
                                              frustum_size=(256, 256, 300))

  with tf.Session() as sess:
    rerendering_np = sess.run(rerendering)
  rerendering_image, rerendering_alpha = \
    rerendering_np[..., :3], rerendering_np[..., [3]]

  background_color = 78.4/100.
  final_composite = background_color*(1-rerendering_alpha)*(1-ground_alpha_np) + \
                    ground_image_np*(1-rerendering_alpha)*ground_alpha_np + \
                    rerendering_image*rerendering_alpha

  _, ax = plt.subplots(1, 2)
  ax[0].imshow(rerendering_image)
  ax[1].imshow(final_composite)
  plt.show()

  # ============================================================================
  object_rotation_v = example_data['object_rotation']
  object_translation_v = \
    example_data['object_translation'][:, [1, 0, 2]]*BLENDER_SCALE
  object_elevation_v = example_data['object_elevation']

  ground_occupancy = np.zeros((128, 128, 128, 1), dtype=np.float32)
  ground_occupancy[-2, 1:-2, 1:-2, 0] = 1
  ground_voxel_color = np.ones((128, 128, 128, 3), dtype=np.float32)*\
                       np.array(ground_color, dtype=np.float32)
  ground_voxel_color = np.concatenate([ground_voxel_color, ground_occupancy],
                                      axis=-1)

  scene_voxels = object_voxels*(1-ground_occupancy) + \
                 ground_voxel_color*ground_occupancy

  euler_angles_x = np.deg2rad(180-object_rotation_v)*np.array([1, 0, 0],
                                                              dtype=np.float32)
  euler_angles_y = np.deg2rad(90-object_elevation_v)*np.array([0, 1, 0],
                                                              dtype=np.float32)
  translation_vector = \
    (object_translation_v[0]/(DIAMETER*0.5)).astype(np.float32)

  interpolated_voxels = helpers.object_to_world(scene_voxels,
                                                euler_angles_x,
                                                euler_angles_y,
                                                translation_vector)

  color_input, alpha_input = tf.split(interpolated_voxels, [3, 1], axis=-1)
  voxel_img = visual_hull.render(color_input*alpha_input)
  with tf.Session() as sess:
    voxel_img_np, interpolated_voxels_np =\
      sess.run([voxel_img, interpolated_voxels])

  plt.imshow(voxel_img_np[0])
  plt.show()

  # ==============================================================================
  checkpoint_dir = '/cns/li-d/home/krematas/neural_voxel_renderer/logdir/12845188/2/'

  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    vol_placeholder = tf.placeholder(tf.float32,
                                     shape=[None, 128, 128, 128, 4],
                                     name='input_voxels')
    rerender_placeholder = tf.placeholder(tf.float32,
                                          shape=[None, 256, 256, 3],
                                          name='rerender')
    light_placeholder = tf.placeholder(tf.float32,
                                       shape=[None, 3],
                                       name='input_light')
    model = models.neural_voxel_renderer_plus(vol_placeholder,
                                              rerender_placeholder,
                                              light_placeholder)
    predicted_image_logits, = model.outputs
    saver = tf.train.Saver()

  with tf.Session(graph=g) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    feed_dict = {vol_placeholder: interpolated_voxels_np,
                 rerender_placeholder: final_composite[None, ...]*2.-1,
                 light_placeholder: np.array([example_data['light_position']])}
    predictions = sess.run(predicted_image_logits, feed_dict)

  _, ax = plt.subplots(1, 2, figsize=(10, 10))
  ax[0].imshow(predictions.squeeze()*0.5+0.5)
  ax[0].axis('off')
  ax[0].set_title('NVR+ prediction')

  ax[1].imshow(example_data['image'])
  ax[1].axis('off')
  ax[1].set_title('Ground truth')
  plt.show()


if __name__ == '__main__':
  tf.app.run(main)
