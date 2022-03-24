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
"""pix3d dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow_datasets import features as tfds_features
import tensorflow_datasets.public_api as tfds

from tensorflow_graphics.datasets import features as tfg_features

_CITATION = '''
@inproceedings{pix3d,
  title={Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong
  and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and
  Freeman, William T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
'''

_DESCRIPTION = '''
Pix3D is a large-scale dataset of diverse image-shape pairs
with pixel-level 2D-3D alignment. It has wide applications in shape-related
 tasks including reconstruction, retrieval, viewpoint estimation, etc.

Pix3D contains 10,069 2D-3D pairs of 395 distinct 3D shapes, categorised into
nine object categories. Each sample comprises of an image, 3D shape represented
as (non-watertight) triangle mesh and voxel grid, bounding-box,
 segmentation mask, intrinsic and extrinsic camera parameters and 2D and 3D key
 points.

Notes:
  * The object and camera poses are provided with respect to the scene, whereas
    the camera is placed at the origin. Pix3D also provides the features
    `camera/position_with_respect_to_object` and `camera/inplane_rotation`.
    Those values are defined in object coordinates and will reproduce an image
    that is equivalent to the original image under a homography transformation.
    They are defined for viewer-centered algorithms whose predictions need to be
    rotated back to the canonical view for evaluations against ground truth
    shapes. This is necessary as most algorithms assume that the camera is
    looking at the object's center, the raw input images are usually cropped or
    transformed before sending into their pipeline.
  * There are two wrong segmentation masks in the annotations of the original
    Pix3D dataset (See https://github.com/xingyuansun/pix3d/issues/18 for
    details). We ignore those samples in this version of the dataset. However,
    if you want to use them, we provide own rendered segmentation masks in
    `tensorflow_graphics/datasets/pix3d/fixed_masks/`. Feel free to copy those
    two masks to your local Pix3D directory in `<PIX3D_HOME>/mask/table/`.
    Additionally, you need to add the indices of these samples to the split
    files located at
    `<TF Graphics Repository>/tensorflow_graphics/datasets/pix3d/splits`.
    The index `7953` needs to be appended to the train index and `9657` belongs
    to the test index.

Train/Test split:
  Pix3D does not provide a standard train/test split. Therefore, this
  implementation adopts the S2 split from Mesh R-CNN
   (https://arxiv.org/abs/1906.02739, Sec. 4.2). This split ensures that the 3D
  models appearing in the train and test sets are disjoint.
'''


class Pix3d(tfds.core.GeneratorBasedBuilder):
  """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment."""

  VERSION = tfds.core.Version('0.1.0')

  TRAIN_SPLIT_IDX = os.path.join(os.path.dirname(__file__),
                                 'splits/pix3d_train.npy')
  TEST_SPLIT_IDX = os.path.join(os.path.dirname(__file__),
                                'splits/pix3d_test.npy')

  CLASS_INDEX = ['background', 'bed', 'bookcase', 'chair', 'desk', 'misc',
                 'sofa', 'table', 'tool', 'wardrobe']

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'image': tfds_features.Image(shape=(None, None, 3), dtype=tf.uint8),
            'image/filename': tfds_features.Text(),
            'image/source': tfds_features.Text(),
            '2d_keypoints': tfds_features.FeaturesDict({
                'num_annotators': tf.uint8,
                'num_keypoints': tf.uint8,
                'keypoints': tfds_features.Tensor(shape=(None,),
                                                  dtype=tf.float32),
            }),
            'mask': tfds_features.Image(shape=(None, None, 1), dtype=tf.uint8),
            'model': tfg_features.TriangleMesh(),
            'model/source': tfds_features.Text(),
            '3d_keypoints': tfds_features.Tensor(shape=(None, 3),
                                                 dtype=tf.float32),
            'voxel': tfg_features.VoxelGrid(shape=(128, 128, 128)),
            'pose': tfg_features.Pose(),  # pose of object w.r.t to world.
            'camera': tfds_features.FeaturesDict({
                'parameters': tfg_features.Camera(),
                'position_with_respect_to_object': tfds_features.Tensor(
                    shape=(3,), dtype=tf.float32
                ),
                'inplane_rotation': tf.float32,
            }),
            'category': tfds_features.ClassLabel(
                num_classes=len(self.CLASS_INDEX)
            ),
            'bbox': tfds_features.BBoxFeature(),
            'truncated': tf.bool,
            'occluded': tf.bool,
            'slightly_occluded': tf.bool,
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=None,
        # Homepage of the dataset for documentation
        homepage='http://pix3d.csail.mit.edu/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    pix3d_dir = dl_manager.download_and_extract(
        'http://pix3d.csail.mit.edu/data/pix3d.zip')

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'samples_directory': pix3d_dir,
                'split_file': self.TRAIN_SPLIT_IDX
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'samples_directory': pix3d_dir,
                'split_file': self.TEST_SPLIT_IDX
            },
        ),
    ]

  def _generate_examples(self, samples_directory, split_file):
    """Yields examples.

    As Pix3D does not come with a predefined train/test split, we adopt one from
    Mesh R-CNN. The split ensures
    that the 3D models appearing in the train and test sets are disjoint.

    Args:
      samples_directory: `str`, path to the directory where Pix3D is stored.
      split_file: `str`, path to .npy file containing the indices of the current
      split.
    """

    with tf.io.gfile.GFile(os.path.join(samples_directory, 'pix3d.json'),
                           mode='r') as pix3d_index:
      pix3d = json.load(pix3d_index)

    split_samples = map(pix3d.__getitem__, np.load(split_file))

    def _build_bbox(box, img_size):
      """Create a BBox with correct order of coordinates.

      Args:
        box: Bounding box of the object as provided by Pix3d
        img_size:  size of the image, in the format of [width, height]

      Returns:
        tfds.features.BBox.
      """
      xmin, ymin, xmax, ymax = box
      width, height = img_size
      return tfds_features.BBox(ymin=ymin / height, xmin=xmin / width,
                                ymax=ymax / height, xmax=xmax / width)

    def _build_camera(f, img_size):
      """Prepare features for `Camera` FeatureConnector.

      The pose originates from the official Pix3D GitHub repository and
      describes the cameras position with respect to the scene.

      The focal length is originally provided in mm, but will be converted to
      pixel here using the fixed sensor with of 32 mm, which also originates
      from the Pix3D GitHub repository.

      Link to the official Pix3D repository:
      https://github.com/xingyuansun/pix3d.

      Args:
        f: float, denoting the focal length in mm.
        img_size: tuple of two floats, denoting the image height and width.

      Returns:
        Dictionary with all Camera Parameters.
      """
      sensor_width = 32.
      return {
          'pose': {
              'R': np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
                            dtype=np.float32),
              't': np.zeros(3, dtype=np.float32)
          },
          'optical_center': (img_size[0] / 2, img_size[1] / 2),
          'f': (f / sensor_width * img_size[0])
      }

    def _build_2d_keypoints(keypoints):
      """Wraps keypoint feature in dict, because TFDS does not allow more than sone unknown dimensions.

      Args:
        keypoints: Array of dimension `[N, M, 2]`, where N denotes the number of
          annotators and M is the number of 2D keypoints. Keypoints are stored
          as (origin: top left corner; +x: rightward; +y: downward);
          [-1.0, -1.0] if an annotator marked this keypoint hard to label.

      Returns:
        Dictionary containing shape and flattened keypoints.
      """
      if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError('2D keypoints should be in shape (N, M, 2).')

      return {
          'num_annotators': keypoints.shape[0],
          'num_keypoints': keypoints.shape[1],
          'keypoints': keypoints.ravel()
      }

    for sample in split_samples:
      example = {
          'image': os.path.join(samples_directory, sample['img']),
          'image/filename': sample['img'],
          'image/source': sample['img_source'],
          '2d_keypoints': _build_2d_keypoints(
              np.asarray(sample['2d_keypoints'],
                         dtype=np.float32)),
          'mask': os.path.join(samples_directory, sample['mask']),
          'model': os.path.join(samples_directory, sample['model']),
          'model/source': sample['model_source'],
          '3d_keypoints': np.loadtxt(
              os.path.join(samples_directory, sample['3d_keypoints']),
              dtype=np.float32),
          'voxel': {
              'path': os.path.join(samples_directory, sample['voxel']),
              'key': 'voxel'
          },
          'pose': {
              'R': sample['rot_mat'],
              't': sample['trans_mat']
          },
          'camera': {
              'parameters': _build_camera(
                  sample['focal_length'],
                  sample['img_size'],
              ),
              'position_with_respect_to_object': sample['cam_position'],
              'inplane_rotation': sample['inplane_rotation'],
          },
          'category': self.CLASS_INDEX.index(sample['category']),
          'bbox': _build_bbox(sample['bbox'], sample['img_size']),
          'truncated': sample['truncated'],
          'occluded': sample['occluded'],
          'slightly_occluded': sample['slightly_occluded']
      }

      yield sample['img'], example
