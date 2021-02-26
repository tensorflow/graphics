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
"""Training procedure for occluded parts."""

import os
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_graphics.projects.points_to_3Dobjects.data_preparation.extract_protos as extract_protos
from tensorflow_graphics.projects.points_to_3Dobjects.losses import collision_loss
from tensorflow_graphics.projects.points_to_3Dobjects.losses import cross_entropy_loss
from tensorflow_graphics.projects.points_to_3Dobjects.losses import focal_loss
from tensorflow_graphics.projects.points_to_3Dobjects.losses import focal_loss_sparse
from tensorflow_graphics.projects.points_to_3Dobjects.losses import learning_rate_schedule
from tensorflow_graphics.projects.points_to_3Dobjects.losses import regression_huber_loss
from tensorflow_graphics.projects.points_to_3Dobjects.losses import regression_l1_loss
import tensorflow_graphics.projects.points_to_3Dobjects.models.centernet_vid as centernet_vid
from tensorflow_graphics.projects.points_to_3Dobjects.transforms import transforms_factory
from tensorflow_graphics.projects.points_to_3Dobjects.utils import evaluator as evaluator_util
import tensorflow_graphics.projects.points_to_3Dobjects.utils.io as io
import tensorflow_graphics.projects.points_to_3Dobjects.utils.logger as logger_util
import tensorflow_graphics.projects.points_to_3Dobjects.utils.plot as plot
import tensorflow_graphics.projects.points_to_3Dobjects.utils.tf_utils as tf_utils

from google3.pyglib import gfile

LOG_DIR = '/occluded_primitives/logs/'
TFRECORDS_DIR = '/occluded_primitives/data_stefan/'
SHAPENET_DIR = '/occluded_primitives/shapenet/'

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', LOG_DIR, 'Path to log directory.')
flags.DEFINE_string('tfrecords_dir', TFRECORDS_DIR, 'Path to tfrecord files.')
flags.DEFINE_string('shapenet_dir', SHAPENET_DIR, 'Path to shapenet data.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_string('model', 'hourglass', 'Feature backbone model.')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs.')
flags.DEFINE_integer('n_tfrecords', 100, 'Number of sharded tf records.')
flags.DEFINE_integer('max_num_objects', 3,
                     'Maximum number of objects in the test scene.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('num_classes', 6, 'Number of classes to detect.')
flags.DEFINE_integer('image_width', 256, 'Number of classes to detect.')
flags.DEFINE_integer('image_height', 256, 'Number of classes to detect.')
flags.DEFINE_integer('num_overfitting_samples', 10, 'Overfitting samples.')
flags.DEFINE_string('master', 'local', 'Location of the session.')
flags.DEFINE_boolean('replication', False, 'Store checkpoint with replication.')
flags.DEFINE_string('xmanager_metric', 'metrics/3D_IoU',
                    'Name of the metric to report to XManager')
flags.DEFINE_boolean('run_graph', False, 'Run in Graph mode.')
flags.DEFINE_integer('num_workers', 20, 'Number of parallel preprocessors.')
flags.DEFINE_boolean('train', False, 'Run script in train mode.')
flags.DEFINE_boolean('val', False, 'Run script in validation mode.')
flags.DEFINE_boolean('francis', False, 'Run script in validation mode.')
flags.DEFINE_boolean('plot', True, 'Plot predictions in matplotlib.')
flags.DEFINE_boolean('debug', False, 'Do only 5 iterations per epoch.')
flags.DEFINE_boolean('eval_only', False, 'Run eval at least once.')
flags.DEFINE_boolean('record_val_losses', True, 'Run eval at least once.')
flags.DEFINE_boolean('local_plot_3d', False, 'should we plotisual debugging?')
flags.DEFINE_boolean('qualitative', False, 'plot qualitative results?')
flags.DEFINE_boolean('gaussian_augmentation', False,
                     'plot qualitative results?')
flags.DEFINE_boolean('translation_augmentation', False,
                     'plot qualitative results?')
flags.DEFINE_boolean('rotation_svd', True, 'Regularize rotation using SVD.')
flags.DEFINE_boolean('soft_shape_labels', False, 'Soft shape labels.')
flags.DEFINE_float('soft_shape_labels_a', 0.03, 'Soft shape labels.')
flags.DEFINE_boolean('predict_2d_box', True, 'Predict 2d boudning box?')
flags.DEFINE_float('gradient_clipping_norm', 100.0, 'Clip gradients.')
flags.DEFINE_float('score_threshold', 0.0,
                   'Minimum threshold for valid detection.')
flags.DEFINE_float('polynomial_degree', 3, 'Clip gradients.')
flags.DEFINE_float('label_smoothing', 0.0, 'How much label smoothing?.')

flags.DEFINE_float('sizes_3d_weight', 1.0, 'How much label smoothing?.')
flags.DEFINE_float('shapes_weight', 1.0, 'How much label smoothing?.')
flags.DEFINE_boolean('shape_focal_loss', False, 'Use focal loss for shape?')

flags.DEFINE_float('rotations_3d_weight', -1.0, 'How much label smoothing?.')
flags.DEFINE_float('translations_3d_weight', -1.0, 'How much label smoothing?.')
flags.DEFINE_float('shape_pc_sdf_weight', -1.0, 'How much label smoothing?.')

flags.DEFINE_float('pose_pc_pc_weight', -1.0, 'How much label smoothing?.')
flags.DEFINE_float('projected_pose_pc_pc_weight', -1.0,
                   'How much label smoothing?.')
flags.DEFINE_float('collision_weight', -1.0, 'How much label smoothing?.')
flags.DEFINE_float('shape_sdf_sdf_weight', -1.0, 'How much label smoothing?.')

flags.DEFINE_float('beta', 10, 'Beta in softargmax formulation.')
flags.DEFINE_float('tol', 0.04, 'sdf tolerance')
flags.DEFINE_string('split', 'val', 'Evaluation split {val|test}.')
flags.DEFINE_integer('part_id', -2, 'WHich part of the split?')
flags.DEFINE_integer('number_hourglasses', 1, 'Number of hourglasses.')
flags.DEFINE_string('kernel_regularization', '', 'Evaluation split {val|test}.')
flags.DEFINE_string('continue_from_checkpoint', '', 'Starting checkpoint.')
flags.DEFINE_string('metrics_dir', 'metrics', 'dir with metrics.')
flags.DEFINE_string('qualidir', '/usr/local/google/home/engelmann', 'path')


def get_shapes(scannet=False):
  """Get the shapes."""
  cluster_filepath = os.path.join(FLAGS.tfrecords_dir,
                                  'dict_clusterCenter_class_nearestModel.pkl')
  with gfile.Open(cluster_filepath, 'rb') as file:
    dict_clusters = pickle.load(file)

  shape_centers = []
  shape_sdfs = []
  shape_pointclouds = []
  for _, cluster in sorted(dict_clusters.items()):
    center, class_id, model_name = cluster
    path_prefix = os.path.join(FLAGS.shapenet_dir, class_id, model_name)
    file_sdf = os.path.join(path_prefix, 'model_normalized_sdf.npy')
    file_pointcloud = os.path.join(path_prefix, 'model_normalized_points.npy')
    with gfile.Open(file_sdf, 'rb') as f:
      sdf = np.load(f).astype(np.float32)
    with gfile.Open(file_pointcloud, 'rb') as f:
      pointcloud = np.load(f).astype(np.float32)
    if scannet:
      rot = np.reshape(np.array([0, 0, 1,
                                 0, 1, 0,
                                 -1, 0, 0],
                                dtype=np.float32), [3, 3])
      # rot = np.reshape(np.array([0, 1, 0, -1, 0, 0, 0, 0, 1],
      #                           dtype=np.float32), [3, 3])
      # rot = np.reshape(np.array([0, -1, 0, -1, 0, 0, 0, 0, 1],
      #                           dtype=np.float32), [3, 3])
      pointcloud = np.transpose(rot @ np.transpose(pointcloud))
    shape_centers.append(np.reshape(center, [1, 32, 32, 32]))
    shape_sdfs.append(np.reshape(sdf, [1, 32, 32, 32]))
    shape_pointclouds.append(np.reshape(pointcloud, [1, -1, 3]))

  shape_centers = np.concatenate(shape_centers, axis=0)
  shape_sdfs = np.concatenate(shape_sdfs, axis=0)
  shape_pointclouds = np.concatenate(shape_pointclouds, axis=0)
  return shape_centers, shape_sdfs, shape_pointclouds, dict_clusters


def get_soft_shape_labels(sdfs):
  """Get soft shape labels."""
  num_shapes = sdfs.shape[0]
  pointcloud_distances = np.zeros([num_shapes, num_shapes])

  for i in range(num_shapes):
    for j in range(i + 1, num_shapes):
      dist = np.mean((sdfs[i] - sdfs[j])**2)
      pointcloud_distances[i, j] = dist

  pc_distances = pointcloud_distances + pointcloud_distances.T
  soft_shape_labels = tf.cast(
      tf.math.less(pc_distances, FLAGS.soft_shape_labels_a + 0.00001),
      dtype=tf.float32)
  soft_shape_labels = \
      tf.maximum(1 - (pc_distances*FLAGS.soft_shape_labels_a)**2, 0.0)

  return soft_shape_labels


def get_model(shape_centers, shape_sdfs, shape_pointclouds, dict_clusters):
  """Get model."""
  kernel_regularization = None
  if FLAGS.kernel_regularization == 'l2' or FLAGS.kernel_regularization == 'l1':
    kernel_regularization = FLAGS.kernel_regularization

  head_losses = {}
  head_losses['centers'] = {'loss': focal_loss.FocalLoss(), 'weight': 1.0}
  if FLAGS.predict_2d_box:
    head_losses['offset'] = {'loss': regression_l1_loss.RegL1Loss(),
                             'weight': 1.0}
    head_losses['width_height'] = {'loss': regression_l1_loss.RegL1Loss(),
                                   'weight': 0.1}
  head_losses['sizes_3d'] = {'loss': regression_l1_loss.RegL1Loss(sparse=True),
                             'weight': FLAGS.sizes_3d_weight}

  if FLAGS.shapes_weight >= 0.0:
    if FLAGS.shape_focal_loss:
      head_losses['shapes'] = {'loss': focal_loss_sparse.SparseFocalLoss(),
                               'weight': FLAGS.shapes_weight}
    else:
      head_losses['shapes'] = \
          {'loss': cross_entropy_loss.CrossEntropyLoss(
              label_smoothing=FLAGS.label_smoothing,
              soft_shape_labels=FLAGS.soft_shape_labels),
           'weight': FLAGS.shapes_weight}
  if FLAGS.translations_3d_weight >= 0.0:
    head_losses['translations_3d'] = \
        {'loss': regression_l1_loss.RegL1Loss(sparse=True),
         'weight': FLAGS.translations_3d_weight}
  if FLAGS.rotations_3d_weight >= 0.0:
    head_losses['rotations_3d'] = \
        {'loss': regression_l1_loss.RegL1Loss(sparse=True),
         'weight': FLAGS.rotations_3d_weight}
  if FLAGS.pose_pc_pc_weight >= 0.0:
    head_losses['pose_groundtruth_pointclouds'] = \
        {'loss': regression_huber_loss.HuberLoss(),
         'weight': FLAGS.pose_pc_pc_weight}
  if FLAGS.projected_pose_pc_pc_weight >= 0.0:
    head_losses['projected_gt_shapes'] = \
        {'loss': regression_huber_loss.HuberLoss(),
         'weight': FLAGS.projected_pose_pc_pc_weight}
  if FLAGS.collision_weight >= 0.0:
    head_losses['collisions_gt_shapes'] = \
        {'loss': collision_loss.CollisionLoss(tol=-FLAGS.tol),
         'weight': FLAGS.collision_weight}
  if FLAGS.shape_sdf_sdf_weight >= 0.0:
    head_losses['sdfs'] = {'loss': regression_huber_loss.HuberLoss(),
                           'weight': FLAGS.shape_sdf_sdf_weight}

  num_shapes = 300
  if 'scannet' in FLAGS.tfrecords_dir:
    num_shapes = 50 * 8

  model = centernet_vid.CenterNetVID(
      heads_losses=head_losses,
      heads={'centers': {'dim': FLAGS.num_classes},
             'offset': {'dim': 2},
             'width_height': {'dim': 2},
             'sizes_offset_3d': {'dim': 3},
             'translations_offset_3d': {'dim': 3},
             'rotations_offset_3d': {'dim': 9},
             'shapes': {'dim': num_shapes}
             },
      input_shape=(None, FLAGS.image_height, FLAGS.image_width, 3),
      get_k_predictions_test=FLAGS.max_num_objects,
      score_threshold=FLAGS.score_threshold,
      number_hourglasses=FLAGS.number_hourglasses,
      kernel_regularization=kernel_regularization,
      clip_norm=FLAGS.gradient_clipping_norm,
      shape_centers=shape_centers,
      shape_sdfs=shape_sdfs,
      shape_pointclouds=shape_pointclouds,
      dict_clusters=dict_clusters,
      beta=FLAGS.beta,
      rotation_svd=FLAGS.rotation_svd,
  )
  model.batch_size = FLAGS.batch_size
  model.window_size = 1
  return model


def get_dataset(split, shape_soft_labels, shape_pointclouds=None):
  """Get dataset."""
  if shape_pointclouds:
    print(shape_pointclouds)
  tfrecord_path = os.path.join(FLAGS.tfrecords_dir, split)
  buffer_size, shuffle, cycles = 10000, True, 10000
  if FLAGS.debug:
    buffer_size, shuffle, cycles = 1, False, 1
  if FLAGS.val:
    buffer_size, shuffle, cycles = 100, False, 1

  tfrecords_pattern = io.expand_rio_pattern(tfrecord_path)
  dataset = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=shuffle)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=cycles)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=buffer_size)

  if 'scannet' in tfrecord_path:
    dataset = dataset.map(extract_protos.decode_bytes_multiple_scannet)
    dataset = dataset.filter(lambda sample: sample['num_boxes'] == 3)
  else:
    dataset = dataset.map(extract_protos.decode_bytes_multiple)
    dataset = \
        dataset.filter(lambda sample: tf.reduce_min(sample['shapes']) > -1)

  def augment(sample):
    image = sample['image']
    if tf.random.uniform([1], 0, 1.0) < 0.8:
      image = tf.image.random_saturation(image, 1.0, 10.0)
      image = tf.image.random_contrast(image, 0.05, 5.0)
      image = tf.image.random_hue(image, 0.5)
      image = tf.image.random_brightness(image, 0.8)
      sample['image'] = image
    if tf.random.uniform([1], 0, 1.0) < 0.5:
      sample['image'] = tf.image.flip_left_right(sample['image'])
      sample['translations_3d'] *= [[-1.0, 1.0, 1.0]]
      sample['rotations_3d'] = tf.reshape(sample['rotations_3d'], [-1, 3, 3])
      sample['rotations_3d'] = tf.transpose(sample['rotations_3d'],
                                            perm=[0, 2, 1])
      sample['rotations_3d'] = tf.reshape(sample['rotations_3d'], [-1, 9])
      bbox = sample['groundtruth_boxes']
      bbox = tf.stack([bbox[:, 0], 1 - bbox[:, 3], bbox[:, 2], 1 - bbox[:, 1]],
                      axis=-1)
      sample['groundtruth_boxes'] = bbox
    if FLAGS.gaussian_augmentation:
      if tf.random.uniform([1], 0, 1.0) < 0.15:
        sample['image'] = tf_utils.gaussian_blur(sample['image'], sigma=1)
      elif tf.random.uniform([1], 0, 1.0) < 0.30:
        sample['image'] = tf_utils.gaussian_blur(sample['image'], sigma=2)
      elif tf.random.uniform([1], 0, 1.0) < 0.45:
        sample['image'] = tf_utils.gaussian_blur(sample['image'], sigma=3)
    return sample

  if FLAGS.train:  # and not FLAGS.debug:
    dataset = dataset.map(augment, num_parallel_calls=FLAGS.num_workers)

  def add_soft_shape_labels(sample):
    sample['shapes_soft'] = tf.map_fn(
        fn=lambda t: tf.cast(shape_soft_labels[t], tf.float32),
        elems=tf.cast(sample['shapes'], tf.int32),
        fn_output_signature=tf.float32)
    return sample

  if FLAGS.soft_shape_labels:
    dataset = dataset.map(add_soft_shape_labels,
                          num_parallel_calls=FLAGS.num_workers)

  # Create dataset for overfitting when debugging
  if FLAGS.debug:
    t = FLAGS.num_overfitting_samples
    dataset = dataset.take(t)
    mult = 1 if FLAGS.val else 1000
    dataset = dataset.repeat(mult)
    if t > 1:
      dataset = dataset.shuffle(buffer_size)

  return dataset


def get_learning_rate_fn():
  decay_steps = int(8e6)
  start_decay_step = int(30.0e6)
  warmup_steps = int(150000)
  if FLAGS.debug:
    decay_steps = 30000
    start_decay_step = 20000
    warmup_steps = 0
  return learning_rate_schedule.WarmupDelayedCosineDecay(
      initial_learning_rate=tf.cast(0.0001, dtype=tf.float32),
      constant_learning_rate=tf.cast(FLAGS.learning_rate, dtype=tf.float32),
      end_learning_rate=FLAGS.learning_rate / 1000000.0,
      warmup_steps=warmup_steps,
      start_decay_step=start_decay_step,
      decay_steps=decay_steps,
  )


def get_evaluator():
  """Evaluator."""
  evaluator = evaluator_util.Evaluator({
      # 'shape_accuracy_topk1': evaluator_util.ShapeAccuracyMetric(k=1),
      # 'shape_accuracy_topk10': evaluator_util.ShapeAccuracyMetric(k=10),
      # 'shape_accuracy_topk100': evaluator_util.ShapeAccuracyMetric(k=100),
      '2D_mAP_25': evaluator_util.BoxIoUMetric(t=0.25, threed=False),
      '2D_mAP_50': evaluator_util.BoxIoUMetric(t=0.50, threed=False),
      # '2D_mAP_60': evaluator_util.BoxIoUMetric(t=0.60, threed=False),
      # '2D_mAP_75': evaluator_util.BoxIoUMetric(t=0.75, threed=False),
      # '2D_mAP_80': evaluator_util.BoxIoUMetric(t=0.80, threed=False),
      # '2D_mAP_90': evaluator_util.BoxIoUMetric(t=0.90, threed=False),
      '3D_mAP_25': evaluator_util.BoxIoUMetric(t=0.25, threed=True),
      '3D_mAP_50': evaluator_util.BoxIoUMetric(t=0.50, threed=True),
      '3D_IoU': evaluator_util.IoUMetric(max_num_classes=FLAGS.num_classes,
                                         tol=FLAGS.tol, resolution=64),
      # 'Collision': evaluator_util.CollisionMetric(tol=-FLAGS.tol),
      # '3D_mAP_60': evaluator_util.BoxIoUMetric(t=0.60, threed=True),
      # '3D_mAP_75': evaluator_util.BoxIoUMetric(t=0.75, threed=True),
      # '3D_mAP_80': evaluator_util.BoxIoUMetric(t=0.80, threed=True),
      # '3D_mAP_90': evaluator_util.BoxIoUMetric(t=0.90, threed=True)
      }, split=FLAGS.split, shapenet_dir=FLAGS.shapenet_dir)
  slave = (FLAGS.part_id > -1)
  if slave:
    iou_path = os.path.join(FLAGS.logdir,
                            FLAGS.metrics_dir, 'iou',
                            'iou_'+str(FLAGS.part_id).zfill(4)+'.pkl')
    if not tf.io.gfile.exists(os.path.dirname(iou_path)):
      tf.io.gfile.makedirs(os.path.dirname(iou_path))

    collision_path = os.path.join(
        FLAGS.logdir,
        FLAGS.metrics_dir,
        'collision',
        'collision_'+str(FLAGS.part_id).zfill(4)+'.pkl')
    if not tf.io.gfile.exists(os.path.dirname(collision_path)):
      tf.io.gfile.makedirs(os.path.dirname(collision_path))

    evaluator = evaluator_util.Evaluator(
        {'3D_IoU': evaluator_util.IoUMetric(max_num_classes=FLAGS.num_classes,
                                            slave=slave, path=iou_path,
                                            tol=FLAGS.tol),
        }, split=FLAGS.split, shapenet_dir=FLAGS.shapenet_dir)
  if FLAGS.francis:
    evaluator = evaluator_util.Evaluator({},
                                         split=FLAGS.split,
                                         shapenet_dir=FLAGS.shapenet_dir)
  return evaluator


def _train_epoch(epoch, model: centernet_vid.CenterNetVID, dataset, logger,
                 number_of_steps_previous_epochs, max_num_steps_epoch,
                 input_image_size):
  """Train one epoch."""
  strategy = tf.distribute.get_strategy()

  if input_image_size:
    print(input_image_size)

  def distributed_step(sample):
    outputs = {}
    logging.info(sample['name'])
    per_replica_outputs = strategy.run(model.train_sample, args=(sample,))
    losses_value = {}
    for key, value in per_replica_outputs.items():
      losses_value[key] = strategy.reduce(
          tf.distribute.ReduceOp.SUM, value, axis=None)

    losses_val_value = {'total_loss': tf.constant(0, dtype=tf.float32)}
    total_loss_diff = tf.abs(losses_value['total_loss'] -
                             losses_val_value['total_loss'])
    return {**losses_value, 'total_loss_diff': total_loss_diff}, outputs

  if FLAGS.run_graph:
    distributed_step = tf.function(distributed_step)

  logger.reset_losses()
  dataset_iterator = iter(dataset)

  n_steps = 0
  while True:
    start_time = time.time()
    sample = tf_utils.get_next_sample_dataset(dataset_iterator)

    if (sample is None or
        max_num_steps_epoch is not None and n_steps > max_num_steps_epoch):
      break
    n_steps += FLAGS.batch_size

    total_num_steps = n_steps + number_of_steps_previous_epochs
    logger.record_scalar('meta/time_read', time.time() - start_time,
                         total_num_steps)
    logger.record_scalar('meta/learning_rate',
                         model.optimizer.learning_rate(total_num_steps),
                         total_num_steps)
    start_time = time.time()
    losses_value, _ = distributed_step(sample)
    logging.info('Loss: %f/%f \t LR: %f GD: %f',
                 losses_value['total_loss'].numpy(),
                 losses_value['total_loss_diff'].numpy(),
                 model.optimizer.learning_rate(total_num_steps),
                 losses_value['gradients_norm'].numpy())
    logger.record_scalar('meta/forward_pass',
                         time.time() - start_time,
                         total_num_steps)
    logger.record_losses('iterations/', losses_value, total_num_steps)

  logger.record_losses_epoch('epoch/', epoch)
  return n_steps


def train(max_num_steps_epoch=None,
          save_initial_checkpoint=False,
          gpu_ids=None):
  """Train function."""

  strategy = tf.distribute.MirroredStrategy(tf_utils.get_devices(gpu_ids))
  logging.info('Number of devices: %d', strategy.num_replicas_in_sync)
  shape_centers, shape_sdfs, shape_pointclouds, dict_clusters = \
      get_shapes('scannet' in FLAGS.tfrecords_dir)
  soft_shape_labels = get_soft_shape_labels(shape_sdfs)
  dataset = get_dataset('train*.tfrecord', soft_shape_labels, shape_pointclouds)

  for sample in dataset.take(1):
    plt.imshow(sample['image'])

  if FLAGS.debug:
    FLAGS.num_epochs = 50
  if FLAGS.continue_from_checkpoint:
    FLAGS.num_epochs *= 2

  latest_epoch = tf.Variable(0, trainable=False)
  num_epochs_var = tf.Variable(FLAGS.num_epochs, trainable=False)
  number_of_steps_previous_epochs = tf.Variable(0, trainable=False,
                                                dtype=tf.int64)
  with strategy.scope():
    work_unit = None
    logging_dir = os.path.join(FLAGS.logdir, 'logging')
    logger = logger_util.Logger(logging_dir, 'train', work_unit, '',
                                save_loss_tensorboard_frequency=100,
                                print_loss_frequency=1000)

    optimizer = tf.keras.optimizers.Adam(learning_rate=get_learning_rate_fn())
    model = get_model(shape_centers,
                      shape_sdfs,
                      shape_pointclouds,
                      dict_clusters)
    model.optimizer = optimizer

    transforms = {'name': 'centernet_preprocessing',
                  'params': {'image_size': (FLAGS.image_height,
                                            FLAGS.image_width),
                             'transform_gt_annotations': True,
                             'random': False}}
    train_targets = {'name': 'centernet_train_targets',
                     'params': {'num_classes': FLAGS.num_classes,
                                'image_size': (FLAGS.image_height,
                                               FLAGS.image_width),
                                'stride': model.output_stride}}
    transform_fn = transforms_factory.TransformsFactory.get_transform_group(
        **transforms)
    train_targets_fn = transforms_factory.TransformsFactory.get_transform_group(
        **train_targets)
    input_image_size = transforms['params']['image_size']

    dataset = dataset.map(transform_fn, num_parallel_calls=FLAGS.num_workers)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.map(train_targets_fn,
                          num_parallel_calls=FLAGS.num_workers)

    if FLAGS.batch_size > 1:
      dataset.prefetch(int(FLAGS.batch_size * 1.5))

    # for sample in dataset:
    #   print(sample['name'])

    dataset = strategy.experimental_distribute_dataset(dataset)

    checkpoint_dir = os.path.join(FLAGS.logdir, 'training_ckpts')

    if FLAGS.replication:
      checkpoint_dir = os.path.join(checkpoint_dir, 'r=30')

    checkpoint = tf.train.Checkpoint(
        epoch=latest_epoch,
        model=model.network,
        optimizer=optimizer,
        number_of_steps_previous_epochs=number_of_steps_previous_epochs,
        num_epochs=num_epochs_var)

    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=5)

    # Restore latest checkpoint
    if manager.latest_checkpoint:
      logging.info('Restoring from %s', manager.latest_checkpoint)
      checkpoint.restore(manager.latest_checkpoint)
    elif FLAGS.continue_from_checkpoint:
      init_checkpoint_dir = os.path.join(
          FLAGS.continue_from_checkpoint, 'training_ckpts')
      init_manager = tf.train.CheckpointManager(checkpoint,
                                                init_checkpoint_dir,
                                                None)
      logging.info('Restoring from pretrained %s',
                   init_manager.latest_checkpoint)
      checkpoint.restore(init_manager.latest_checkpoint)
    else:
      logging.info('Not restoring any previous training checkpoint.')

    if save_initial_checkpoint and not manager.latest_checkpoint:
      # Create a new checkpoint to avoid internal ckpt counter to increment
      tmp_ckpt = tf.train.Checkpoint(epoch=latest_epoch, model=model.network)
      tmp_manager = tf.train.CheckpointManager(tmp_ckpt, checkpoint_dir, None)
      save_path = tmp_manager.save(0)
      logging.info('Saved checkpoint for epoch %d: %s',
                   int(latest_epoch.numpy()), save_path)
    latest_epoch.assign_add(1)

    with logger.summary_writer.as_default():
      for epoch in range(int(latest_epoch.numpy()), FLAGS.num_epochs + 1):
        latest_epoch.assign(epoch)
        n_steps = _train_epoch(epoch, model, dataset, logger,
                               number_of_steps_previous_epochs,
                               max_num_steps_epoch, input_image_size)
        number_of_steps_previous_epochs.assign_add(n_steps)
        save_path = manager.save()
        logging.info('Saved checkpoint for epoch %d: %s',
                     int(latest_epoch.numpy()), save_path)


def _val_epoch(
    name,
    model,
    dataset,
    input_image_size,
    epoch,
    logger,
    number_of_steps_previous_epochs,
    evaluator: evaluator_util.Evaluator,
    record_loss=False):
  """Validation epoch."""

  if name:
    print(name)

  if FLAGS.part_id > -2:
    record_loss = False
  strategy = tf.distribute.get_strategy()

  def distributed_step(sample):
    training = False
    output, loss = strategy.run(model.test_sample,
                                args=(sample, record_loss, training))
    losses_value = {}
    if record_loss:
      for key, value in loss.items():
        losses_value[key] = strategy.reduce(
            tf.distribute.ReduceOp.SUM, value, axis=None)
    return output, losses_value

  if FLAGS.run_graph:
    distributed_step = tf.function(distributed_step)

  logger.reset_losses()
  evaluator.reset_metrics()

  dataset_iterator = iter(dataset)
  n_steps = tf.constant(0, dtype=tf.int64)

  while True:
    logging.info('val %d', int(n_steps.numpy()))
    start_time = time.time()
    sample = tf_utils.get_next_sample_dataset(dataset_iterator)
    if sample is None or tf_utils.compute_batch_size(sample) == 0:
      break
    n_steps += tf.cast(tf_utils.compute_batch_size(sample), tf.int64)
    logger.record_scalar('meta/time_read',
                         time.time() - start_time,
                         n_steps + number_of_steps_previous_epochs)

    start_time = time.time()
    outputs, losses = distributed_step(sample)
    logger.record_scalar('meta/forward_pass', time.time() - start_time,
                         n_steps + number_of_steps_previous_epochs)
    status = False
    if status:
      model_path = '/usr/local/google/home/engelmann/saved_model'
      model.network.save(model_path, save_format='tf')
      new_model = tf.keras.models.load_model(model_path)
      new_model.summary()

    start_time = time.time()
    if record_loss:
      logger.record_losses('iterations/', losses,
                           n_steps + number_of_steps_previous_epochs)
    outputs = outputs[-1]  # only take outputs from last hourglass
    batch_id = 0

    # We assume batch_size=1 here.
    detections = model.postprocess_sample2(input_image_size, sample, outputs)

    logger.record_scalar('meta/post_processing',
                         time.time() - start_time,
                         n_steps + number_of_steps_previous_epochs)
    tmp_sample = {k: v[0] for k, v in sample.items()}
    result_dict = evaluator.add_detections(tmp_sample, detections)
    iou_mean, iou_min = result_dict['iou_mean'], result_dict['iou_min']

    if (FLAGS.master == 'local' or FLAGS.plot) and not FLAGS.francis and \
      n_steps < tf.constant(13, dtype=tf.int64) and FLAGS.part_id < -1:

      # Plot 3D
      if FLAGS.local_plot_3d:
        logdir = os.path.join(
            '..', os.path.join(*(FLAGS.logdir.split(os.path.sep)[5:])),
            'plots3d', str(sample['scene_filename'][batch_id].numpy())[2:-1])
        logging.info(logdir)
        plot.plot_detections_3d(detections, sample, logdir, model.dict_clusters)

      # Plot 2D
      image = tf.io.decode_image(sample['image_data'][batch_id]).numpy()
      figure_heatmaps = plot.plot_to_image(plot.plot_heatmaps(
          image, detections))
      figure_boxes_2d = plot.plot_to_image(plot.plot_boxes_2d(
          image, sample, detections))
      figure_boxes_3d = plot.plot_to_image(plot.plot_boxes_3d(
          image, sample, detections))

      total_steps = n_steps + number_of_steps_previous_epochs
      tf.summary.image('Heatmaps', figure_heatmaps, total_steps)
      tf.summary.image('Boxes 2D', figure_boxes_2d, total_steps)
      tf.summary.image('Boxes 3D', figure_boxes_3d, total_steps)

    if (FLAGS.part_id > -1 and FLAGS.qualitative) or FLAGS.francis or True:
      logdir = FLAGS.logdir

      if FLAGS.francis:
        logdir = os.path.join(FLAGS.qualidir, 'francis')

      path_input = os.path.join(logdir, 'qualitative', 'img')
      path_blender = os.path.join(logdir, 'qualitative', 'blender2')
      path_2d_min = os.path.join(logdir, 'qualitative', 'img_2d_min')
      path_2d_mean = os.path.join(logdir, 'qualitative', 'img_2d_mean')
      path_3d_min = os.path.join(logdir, 'qualitative', 'img_3d_min')
      path_3d_mean = os.path.join(logdir, 'qualitative', 'img_3d_mean')

      tf.io.gfile.makedirs(path_input)
      tf.io.gfile.makedirs(path_blender)
      tf.io.gfile.makedirs(path_2d_min)
      tf.io.gfile.makedirs(path_2d_mean)
      tf.io.gfile.makedirs(path_3d_min)
      tf.io.gfile.makedirs(path_3d_mean)

      scene_name = \
          str(sample['scene_filename'][0].numpy(), 'utf-8').split('.')[0]
      iou_min_str = f'{iou_min:.5f}' if iou_min >= 0 else '0'
      iou_mean_str = f'{iou_mean:.5f}' if iou_mean >= 0 else '0'
      image = tf.io.decode_image(sample['image_data'][batch_id]).numpy()

      # Plot original image
      _ = plt.figure(figsize=(5, 5))
      plt.clf()
      plt.imshow(image)
      filepath_input = os.path.join(path_input, scene_name+'.png')
      with tf.io.gfile.GFile(filepath_input, 'wb') as f:
        plt.savefig(f)

      # Plot image 2D bounding boxes
      plot.plot_boxes_2d(image, sample, detections,
                         groundtruth=(not FLAGS.francis))
      filepath_2d_min = \
          os.path.join(path_2d_min, iou_min_str+'_'+scene_name+'.png')
      filepath_2d_mean = \
          os.path.join(path_2d_mean, iou_mean_str+'_'+scene_name+'.png')
      for path in [filepath_2d_min, filepath_2d_mean]:
        with tf.io.gfile.GFile(path, 'wb') as f:
          plt.savefig(f)

      # Plot image 3D bounding boxes
      plot.plot_boxes_3d(image,
                         sample,
                         detections,
                         groundtruth=(not FLAGS.francis))
      filepath_3d_min = \
          os.path.join(path_3d_min, iou_min_str+'_'+scene_name+'.png')
      filepath_3d_mean = \
          os.path.join(path_3d_mean, iou_mean_str+'_'+scene_name+'.png')
      for path in [filepath_3d_min, filepath_3d_mean]:
        with tf.io.gfile.GFile(path, 'wb') as f:
          plt.savefig(f)

      if FLAGS.local_plot_3d:
        # Plot 3D visualizer
        path = os.path.join(
            '..', os.path.join(*(logdir.split(os.path.sep)[6:])),
            'qualitative', 'web_3d_min', iou_min_str+'_'+scene_name)
        plot.plot_detections_3d(detections,
                                sample,
                                path,
                                model.dict_clusters,
                                local=FLAGS.francis)
        path = os.path.join(
            '..', os.path.join(*(logdir.split(os.path.sep)[6:])),
            'qualitative', 'web_3d_mean', iou_mean_str+'_'+scene_name)
        plot.plot_detections_3d(detections,
                                sample,
                                path,
                                model.dict_clusters,
                                local=FLAGS.francis)

      # Save pickels for plotting in blender
      path_blender_file = os.path.join(path_blender, scene_name)
      plot.save_for_blender(detections, sample, path_blender_file,
                            model.dict_clusters, model.shape_pointclouds,
                            local=FLAGS.francis)

  if record_loss:
    logger.record_losses_epoch('epoch/', epoch)

  metrics = evaluator.evaluate()
  if record_loss:
    logger.record_dictionary_scalars('metrics/', metrics, epoch)
  # mAP3Ds = ['3D_mAP_50', '3D_mAP_60', '3D_mAP_70', '3D_mAP_80', '3D_mAP_90']
  # mAP3D = np.mean(np.array([metrics[v] for v in mAP3Ds]))
  # logger.record_scalar('metrics/3D_mAP', mAP3D, epoch)
  # mAP2Ds = ['2D_mAP_50', '2D_mAP_60', '2D_mAP_70', '2D_mAP_80', '2D_mAP_90']
  # mAP2D = np.mean(np.array([metrics[v] for v in mAP2Ds]))
  # logger.record_scalar('metrics/2D_mAP', mAP2D, epoch)
  # else:
  #   stats = dataset.evaluate_evaluator()
  #   logger.record_dictionary_scalars(f'{name}_', stats, epoch)
  return n_steps


def val(gpu_ids=None, record_losses=False, split='val', part_id=-2):
  """Val function."""
  FLAGS.batch_size = 1

  strategy = tf.distribute.MirroredStrategy(tf_utils.get_devices(gpu_ids))
  logging.info('Number of devices: %d', strategy.num_replicas_in_sync)
  shape_centers, shape_sdfs, shape_pointclouds, dict_clusters = \
      get_shapes('scannet' in FLAGS.tfrecords_dir)
  soft_shape_labels = get_soft_shape_labels(shape_sdfs)
  part = '*.tfrecord' if part_id == -2 else \
      '-'+str(part_id).zfill(5)+'-of-00100.tfrecord'
  dataset = get_dataset(split+part, soft_shape_labels, shape_pointclouds)

  # for sample in dataset:
  #   plt.imshow(sample['image'])
  #   plt.savefig('/usr/local/google/home/engelmann/res/'+sample['scene_filename'].numpy().decode()+'.png')

  val_evaluator = get_evaluator()

  with strategy.scope():
    name = 'eval_'+str(split)
    work_unit = None
    logging_dir = os.path.join(FLAGS.logdir, 'logging')
    logger = logger_util.Logger(logging_dir, name, work_unit,
                                FLAGS.xmanager_metric,
                                save_loss_tensorboard_frequency=10,
                                print_loss_frequency=1000)
    epoch = tf.Variable(0, trainable=False)
    latest_epoch = tf.Variable(-1, trainable=False)
    num_epochs = tf.Variable(-1, trainable=False)
    number_of_steps_previous_epochs = \
        tf.Variable(0, trainable=False, dtype=tf.int64)

    model = get_model(shape_centers,
                      shape_sdfs,
                      shape_pointclouds,
                      dict_clusters)

    transforms = {'name': 'centernet_preprocessing',
                  'params': {'image_size': (FLAGS.image_height,
                                            FLAGS.image_width),
                             'transform_gt_annotations': True,
                             'random': False}}
    train_targets = {'name': 'centernet_train_targets',
                     'params': {'num_classes': FLAGS.num_classes,
                                'image_size': (FLAGS.image_height,
                                               FLAGS.image_width),
                                'stride': model.output_stride}}
    transform_fn = transforms_factory.TransformsFactory.get_transform_group(
        **transforms)
    train_targets_fn = transforms_factory.TransformsFactory.get_transform_group(
        **train_targets)
    input_image_size = transforms['params']['image_size']

    dataset = dataset.map(transform_fn, num_parallel_calls=FLAGS.num_workers)
    dataset = dataset.batch(FLAGS.batch_size)

    # for k in ['name', 'scene_filename', 'mesh_names', 'classes', 'image',
    #           'image_data', 'original_image_spatial_shape', 'num_boxes',
    #           'center2d', 'groundtruth_boxes', 'dot', 'sizes_3d',
    #           'translations_3d', 'rotations_3d', 'rt', 'k',
    #           'groundtruth_valid_classes', 'shapes']:
    #   print('---', k)
    #   for i, sample in enumerate(dataset.take(7)):
    #     print(sample[k].shape)
    # train_targets_fn(sample)
    # for i, sample in enumerate(dataset):
    #   print(i)
    #   train_targets_fn(sample)

    if train_targets_fn is not None:
      dataset = dataset.map(train_targets_fn,
                            num_parallel_calls=FLAGS.num_workers)

    if FLAGS.debug and False:
      for d in dataset.take(1):
        image = tf.io.decode_image(d['image_data'][0]).numpy()
        heatmaps = d['centers'][0]
        plot.plot_gt_heatmaps(image, heatmaps)

    if tf.distribute.has_strategy():
      strategy = tf.distribute.get_strategy()
      dataset = strategy.experimental_distribute_dataset(dataset)
      if transforms is not None and input_image_size is None:
        if FLAGS.run_graph:
          FLAGS.run_graph = False
          logging.info('Graph mode has been disable because the input does'
                       'not have constant size.')
        if FLAGS.batch_size > strategy.num_replicas_in_sync:
          raise ValueError('Batch size cannot be bigger than the number of GPUs'
                           ' when the input does not have constant size')

    val_checkpoint_dir = os.path.join(FLAGS.logdir, f'{name}_ckpts')
    val_checkpoint = tf.train.Checkpoint(
        epoch=latest_epoch,
        number_of_steps_previous_epochs=number_of_steps_previous_epochs)
    val_manager = tf.train.CheckpointManager(
        val_checkpoint, val_checkpoint_dir, max_to_keep=1)
    if val_manager.latest_checkpoint:
      val_checkpoint.restore(val_manager.latest_checkpoint)

    train_checkpoint_dir = os.path.join(FLAGS.logdir, 'training_ckpts')
    if FLAGS.replication:
      train_checkpoint_dir = os.path.join(train_checkpoint_dir, 'r=30')

    train_checkpoint = tf.train.Checkpoint(epoch=epoch, model=model.network,
                                           num_epochs=num_epochs)
    latest_checkpoint = ''

    if FLAGS.master == 'local' or FLAGS.plot:
      local_dump = os.path.join(FLAGS.logdir, 'images')
      if not tf.io.gfile.exists(local_dump):
        tf.io.gfile.makedirs(local_dump)

    with logger.summary_writer.as_default():
      while True:
        curr_latest_checkpoint = \
            tf.train.latest_checkpoint(train_checkpoint_dir)
        if (curr_latest_checkpoint is not None and
            latest_checkpoint != curr_latest_checkpoint):
          latest_checkpoint = curr_latest_checkpoint
          train_checkpoint.restore(curr_latest_checkpoint)
          if epoch != latest_epoch or FLAGS.eval_only:
            FLAGS.eval_only = False
            logging.info('Evaluating checkpoint in %s: %s.',
                         name, latest_checkpoint)
            n_steps = _val_epoch(name, model, dataset, input_image_size,
                                 epoch.numpy(), logger,
                                 number_of_steps_previous_epochs,
                                 val_evaluator, record_losses)
            number_of_steps_previous_epochs.assign_add(n_steps)

            latest_epoch.assign(epoch.numpy())
            if part_id < -1:
              val_manager.save()
            else:
              return
        if epoch == num_epochs:
          break
        time.sleep(1)


def main(_):

  if FLAGS.debug:
    FLAGS.split = 'train'
  if FLAGS.francis:
    FLAGS.split = 'francis'
  if 'scannet' in FLAGS.tfrecords_dir:
    FLAGS.num_classes = 8
    FLAGS.image_width = 640
    FLAGS.image_height = 640

  if not tf.io.gfile.exists(FLAGS.logdir):
    tf.io.gfile.makedirs(FLAGS.logdir)
  if FLAGS.train:
    train()
  elif FLAGS.val:
    if FLAGS.part_id == -1:

      def eval_iou():
        metrics_dir = os.path.join(FLAGS.logdir, FLAGS.metrics_dir, 'iou')
        if not tf.io.gfile.exists(metrics_dir):
          tf.io.gfile.makedirs(metrics_dir)

        while len(tf.io.gfile.listdir(metrics_dir)) < 100:
          print('waiting...',
                len(tf.io.gfile.listdir(metrics_dir)), 'out of 100')
          time.sleep(5)

        all_iou_per_class = {}
        for i, iou_file in enumerate(tf.io.gfile.listdir(metrics_dir)):
          logging.info(i)
          iou_file_path = os.path.join(metrics_dir, iou_file)
          with gfile.Open(iou_file_path, 'rb') as filename:
            print(iou_file_path)
            iou_per_class = pickle.load(filename)
            for k, v in iou_per_class.items():
              if k not in all_iou_per_class:
                all_iou_per_class[k] = []
              all_iou_per_class[k] = \
                  all_iou_per_class[k] + [n.numpy() for n in v]

        with gfile.Open(metrics_dir + '.txt', 'wb') as file:
          mean_iou_per_class = {}
          all_iou = []
          class_id_to_name = ['chair', 'sofa', 'table', 'bottle', 'bowl', 'mug']
          for k, v in all_iou_per_class.items():
            mean_iou_per_class[k] = np.mean(v)
            file.write(class_id_to_name[k]+':\t'+
                       str(np.mean(v))+' ('+str(np.std(v))+')\n')
            all_iou = all_iou + v
          per_class_mean = np.mean(list(mean_iou_per_class.values()))
          global_mean = np.mean(all_iou)
          file.write('\nmIoU:\t'+str(per_class_mean))
          file.write('\nglobal IoU:\t'+str(global_mean))

      def eval_collision():
        metrics_dir = os.path.join(FLAGS.logdir, FLAGS.metrics_dir, 'collision')
        if not tf.io.gfile.exists(metrics_dir):
          tf.io.gfile.makedirs(metrics_dir)

        while len(tf.io.gfile.listdir(metrics_dir)) < FLAGS.n_tfrecords:
          time.sleep(5)

        total_collisions = 0
        total_intersections = []
        total_ious = []
        for i, file in enumerate(tf.io.gfile.listdir(metrics_dir)):
          logging.info(i)
          file_path = os.path.join(metrics_dir, file)
          with gfile.Open(file_path, 'rb') as filename:
            collision_data = pickle.load(filename)
            total_collisions += np.sum(collision_data['collisions'])
            total_intersections = \
                total_intersections + collision_data['intersections']
            total_ious = total_ious + collision_data['ious']

        with gfile.Open(metrics_dir+'.txt', 'wb') as file:
          file.write('\ncollisions:\t'+str(total_collisions))
          file.write('\nintersect.:\t'+str(np.mean(total_intersections)))
          file.write('\niou:\t'+str(np.mean(total_ious)))

      eval_iou()
      eval_collision()

      return
    else:
      val(record_losses=FLAGS.record_val_losses,
          split=FLAGS.split,
          part_id=FLAGS.part_id)


if __name__ == '__main__':
  app.run(main)
