"""pix3d dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets import features as tfds_features

from tensorflow_graphics.datasets import features as tfg_features
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d

_CITATION = """@inproceedings{pix3d,
  title={Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and Freeman, William T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
"""

_DESCRIPTION = """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment. 
It has wide applications in shape-related tasks including reconstruction, retrieval, viewpoint estimation, etc.

Pix3D contains 10,069 2D-3D pairs of 395 distinct 3D shapes, categorised into nine object categories. 
Each sample comprises of an image, 3D shape represented as (non-watertight) triangle mesh and voxel grid, 
bounding-box, segmentation mask, intrinsic and extrinsic camera parameters and 2D and 3D key points. 
"""


class Pix3d(tfds.core.GeneratorBasedBuilder):
  """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment."""

  # TODO(pix3d): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  _TRAIN_SPLIT_IDX = "splits/pix3d_train.npy"
  _TEST_SPLIT_IDX = "splits/pix3d_test.npy"

  def _info(self):
    # TODO(pix3d): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=_DESCRIPTION,
      # tfds.features.FeatureConnectors
      features=tfds.features.FeaturesDict({
        'image': tfds_features.Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': tfds_features.Text(),
        'image/source': tfds_features.Text(),
        '2d_keypoints': tfds_features.Tensor(shape=(None, 2), dtype=tf.float32),
        'mask': tfds_features.Image(shape=(None, None, 1), dtype=tf.uint8),
        'model': tfg_features.TriangleMesh(),
        'model/source': tfds_features.Text(),
        '3d_keypoints': tfds_features.Tensor(shape=(None, 3), dtype=tf.float32),
        'voxel': tfg_features.VoxelGrid(shape=(128, 128, 128)),
        'pose': tfg_features.Pose(),
        'camera': tfg_features.Camera(),
        'category': tfds_features.ClassLabel(names=['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool',
                                                    'wardrobe']),
        'bbox': tfds_features.BBoxFeature(),
        'truncated': tf.bool,
        'occluded': tf.bool,
        'slightly_occluded': tf.bool
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
    # TODO(pix3d): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    pix3d_dir = dl_manager.download_and_extract('http://pix3d.csail.mit.edu/data/pix3d.zip')

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          'samples_directory': pix3d_dir,
          'split_file': self._TRAIN_SPLIT_IDX
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          'samples_directory': pix3d_dir,
          'split_file': self._TEST_SPLIT_IDX
        },
      ),
    ]

  def _generate_examples(self, samples_directory, split_file):
    """Yields examples.

    As Pix3D does not come with a predefined train/test split, we adopt one from Mesh R-CNN. The split ensures
    that the 3D models appearing in the train and test sets are disjoint.

    Args:
      samples_directory: `str`, path to the directory where Pix3D is stored.
      split_file: `str`, path to .npy file containing the indices of the current split.
    """

    # TODO(pix3d): Yields (key, example) tuples from the dataset

    with tf.io.gfile.GFile(os.path.join(samples_directory, 'pix3d.json'), mode='r') as pix3d_index:
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
      return tfds_features.BBox(ymin=ymin/height, xmin=xmin/width, ymax=ymax/height, xmax=xmax/width)

    def _build_camera(f, position, rotation, img_size):
      """Prepare features for `Camera` FeatureConnector."""
      position = tf.convert_to_tensor(position, dtype=tf.float32)
      position = tf.expand_dims(position, 0)
      rotation = tf.convert_to_tensor(rotation, dtype=tf.float32)
      rotation = tf.expand_dims(rotation, 0)
      return {
        'R': rotation_matrix_3d.from_axis_angle(-position / np.linalg.norm(position), rotation).numpy().reshape(3, 3),
        't': position.numpy().reshape(3, ),
        'optical_center': (img_size[0] / 2, img_size[1] / 2),
        'f': f
      }

    for sample in split_samples:
      example = {
        'image': os.path.join(samples_directory, sample['img']),
        'image/filename': sample['img'],
        'image/source': sample['img_source'],
        '2d_keypoints': np.asarray(sample['2d_keypoints'], dtype=np.float32).reshape(-1, 2),
        'mask': os.path.join(samples_directory, sample['mask']),
        'model': os.path.join(samples_directory, sample['model']),
        'model/source': sample['model_source'],
        '3d_keypoints': np.loadtxt(os.path.join(samples_directory, sample['3d_keypoints']), dtype=np.float32),
        'voxel': {
          'path': os.path.join(samples_directory, sample['voxel']),
          'key': 'voxel'
        },
        'pose': {
          'R': sample['rot_mat'],
          't': sample['trans_mat']
        },
        'camera': _build_camera(
          sample['focal_length'],
          sample['cam_position'],
          sample['inplane_rotation'],
          sample['img_size'],
        ),
        'category': sample['category'],
        'bbox': _build_bbox(sample['bbox'], sample['img_size']),
        'truncated': sample['truncated'],
        'occluded': sample['occluded'],
        'slightly_occluded': sample['slightly_occluded']
      }

      yield sample['img'], example
