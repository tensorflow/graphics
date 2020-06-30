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
"""Training Loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

  logging.info("=> Starting ...")

  # Select dataset.
  logging.info("=> Preparing datasets ...")
  data = datasets.get_dataset(FLAGS.dataset, "train", FLAGS)
  batch = tf.data.make_one_shot_iterator(data).get_next()

  # Select model.
  logging.info("=> Creating {} model".format(FLAGS.model))
  model = models.get_model(FLAGS.model, FLAGS)
  optimizer = tf.train.AdamOptimizer(FLAGS.lr)

  # Set up the graph
  train_loss, train_op, global_step = model.compute_loss(
      batch, training=True, optimizer=optimizer)

  # Training hooks
  stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.max_steps)
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
  ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100, summary_writer=summary_writer, summary_op=ops)
  step_counter_hook = tf.train.StepCounterHook(summary_writer=summary_writer)
  hooks = [stop_hook, step_counter_hook, summary_hook]

  logging.info("=> Start training loop ...")
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.train_dir,
      hooks=hooks,
      scaffold=None,
      save_checkpoint_steps=FLAGS.save_every,
      save_checkpoint_secs=None,
      save_summaries_steps=None,
      save_summaries_secs=None,
      log_step_count_steps=None,
      max_wait_secs=3600) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run([batch, train_loss, global_step, train_op])


if __name__ == "__main__":
  tf.app.run(main)
