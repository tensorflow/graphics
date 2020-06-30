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
"""Evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.cvxnet.lib import datasets
from tensorflow_graphics.projects.cvxnet.lib import models
from tensorflow_graphics.projects.cvxnet.lib import utils

tf.disable_eager_execution()

flags = tf.app.flags
logging = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)

utils.define_flags()
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.set_random_seed(2191997)
  np.random.seed(6281996)

  logging.info('=> Starting ...')
  eval_dir = path.join(FLAGS.train_dir, 'eval')

  # Select dataset.
  logging.info('=> Preparing datasets ...')
  data = datasets.get_dataset(FLAGS.dataset, 'test', FLAGS)
  batch = tf.data.make_one_shot_iterator(data).get_next()

  # Select model.
  logging.info('=> Creating {} model'.format(FLAGS.model))
  model = models.get_model(FLAGS.model, FLAGS)

  # Set up the graph
  global_step = tf.train.get_or_create_global_step()
  test_loss, test_iou = model.compute_loss(batch, training=False)
  if FLAGS.extract_mesh or FLAGS.surface_metrics:
    img_ch = 3 if FLAGS.image_input else FLAGS.depth_d
    input_holder = tf.placeholder(tf.float32, [None, 224, 224, img_ch])
    params = model.encode(input_holder, training=False)
    params_holder = tf.placeholder(tf.float32, [None, model.n_params])
    points_holder = tf.placeholder(tf.float32, [None, None, FLAGS.dims])
    indicators, unused_var = model.decode(
        params_holder, points_holder, training=False)
  if (not FLAGS.extract_mesh) or (not FLAGS.surface_metrics):
    summary_writer = tf.summary.FileWriter(eval_dir)
    iou_holder = tf.placeholder(tf.float32)
    iou_summary = tf.summary.scalar('test_iou', iou_holder)

  logging.info('=> Evaluating ...')
  last_step = -1
  while True:
    shapenet_stats = utils.init_stats()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[],
        save_checkpoint_steps=None,
        save_checkpoint_secs=None,
        save_summaries_steps=None,
        save_summaries_secs=None,
        log_step_count_steps=None,
        max_wait_secs=3600) as mon_sess:
      step_val = mon_sess.run(global_step)
      if step_val <= last_step:
        continue
      else:
        last_step = step_val
      while not mon_sess.should_stop():
        batch_val, unused_var, test_iou_val = mon_sess.run(
            [batch, test_loss, test_iou])
        if FLAGS.extract_mesh or FLAGS.surface_metrics:
          if FLAGS.image_input:
            input_val = batch_val['image']
          else:
            input_val = batch_val['depth']
          mesh = utils.extract_mesh(
              input_val,
              params,
              indicators,
              input_holder,
              params_holder,
              points_holder,
              mon_sess,
              FLAGS,
          )
          if FLAGS.trans_dir is not None:
            utils.transform_mesh(mesh, batch_val['name'], FLAGS.trans_dir)
        if FLAGS.extract_mesh:
          utils.save_mesh(mesh, batch_val['name'], eval_dir)
        if FLAGS.surface_metrics:
          chamfer, fscore = utils.compute_surface_metrics(
              mesh, batch_val['name'], FLAGS.mesh_dir)
        else:
          chamfer = fscore = 0.
        example_stats = utils.Stats(
            iou=test_iou_val[0], chamfer=chamfer, fscore=fscore)
        utils.update_stats(example_stats, batch_val['name'], shapenet_stats)
    utils.average_stats(shapenet_stats)
    if (not FLAGS.extract_mesh) and (not FLAGS.surface_metrics):
      with tf.Session() as sess:
        iou_summary_val = sess.run(
            iou_summary, feed_dict={iou_holder: shapenet_stats['all']['iou']})
        summary_writer.add_summary(iou_summary_val, step_val)
        summary_writer.flush()
    if FLAGS.surface_metrics:
      utils.write_stats(
          shapenet_stats,
          eval_dir,
          step_val,
      )
    if FLAGS.eval_once or step_val >= FLAGS.max_steps:
      break


if __name__ == '__main__':
  tf.app.run(main)
