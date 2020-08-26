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
"""Reconstruction Evaluation."""
from os import path

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.nasa.lib import datasets
from tensorflow_graphics.projects.nasa.lib import models
from tensorflow_graphics.projects.nasa.lib import utils

tf.disable_eager_execution()

flags = tf.app.flags
logging = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)

utils.define_flags()
FLAGS = flags.FLAGS


def build_eval_graph(input_fn, model_fn, hparams):
  """Build the evaluation computation graph."""
  dataset = input_fn(None)
  batch = dataset.make_one_shot_iterator().get_next()

  batch_holder = {
      "transform":
          tf.placeholder(
              tf.float32,
              [1, 1, hparams.n_parts, hparams.n_dims + 1, hparams.n_dims + 1]),
      "joint":
          tf.placeholder(tf.float32, [1, 1, hparams.n_parts, hparams.n_dims]),
      "point":
          tf.placeholder(tf.float32, [1, 1, None, hparams.n_dims]),
      "label":
          tf.placeholder(tf.float32, [1, 1, None, 1]),
  }
  latent_holder, latent, occ = model_fn(batch_holder, None, None, "gen_mesh")

  # Eval Summary
  iou_holder = tf.placeholder(tf.float32, [])
  best_holder = tf.placeholder(tf.float32, [])
  tf.summary.scalar("IoU", iou_holder)
  tf.summary.scalar("Best_IoU", best_holder)

  return {
      "batch_holder": batch_holder,
      "latent_holder": latent_holder,
      "latent": latent,
      "occ": occ,
      "batch": batch,
      "iou_holder": iou_holder,
      "best_holder": best_holder,
      "merged_summary": tf.summary.merge_all(),
  }


def evaluate(hook_dict, ckpt, saver, best_iou, hparams):
  """Evaluate a checkpoint on the whole test set."""
  batch = hook_dict["batch"]
  merged_summary = hook_dict["merged_summary"]
  iou_holder = hook_dict["iou_holder"]
  best_holder = hook_dict["best_holder"]
  batch_holder = hook_dict["batch_holder"]
  latent_holder = hook_dict["latent_holder"]
  latent = hook_dict["latent"]
  occ = hook_dict["occ"]
  global_step = utils.parse_global_step(ckpt)

  assignment_map = {
      "shape/": "shape/",
  }
  tf.train.init_from_checkpoint(ckpt, assignment_map)
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    accum_iou = 0.
    example_cnt = 0
    while True:
      try:
        batch_val = sess.run(batch)
        feed_dict = {
            batch_holder["transform"]: batch_val["transform"],
            batch_holder["joint"]: batch_val["joint"],
        }

        iou = utils.compute_iou(sess, feed_dict, latent_holder,
                                batch_holder["point"], latent, occ[:, -1:],
                                batch_val["points"], batch_val["labels"],
                                hparams)
        accum_iou += iou
        example_cnt += 1

        if hparams.gen_mesh_only > 0:
          # Generate meshes for evaluation
          unused_var = utils.save_mesh(
              sess,
              feed_dict,
              latent_holder,
              batch_holder["point"],
              latent,
              occ,
              batch_val,
              hparams,
          )
          logging.info("Generated mesh No.{}".format(example_cnt))

      except tf.errors.OutOfRangeError:
        accum_iou /= example_cnt

        if best_iou < accum_iou:
          best_iou = accum_iou
          saver.save(sess, path.join(hparams.train_dir, "best", "model.ckpt"),
                     global_step)
        summary = sess.run(
            merged_summary,
            utils.make_summary_feed_dict(
                iou_holder,
                accum_iou,
                best_holder,
                best_iou,
            ))

        # If only generating meshes for the sequence, we can determinate the
        # evaluation after the first full loop over the test set.
        if hparams.gen_mesh_only:
          exit(0)

        break

  return summary, global_step


def main(unused_argv):
  tf.random.set_random_seed(20200823)
  np.random.seed(20200823)

  input_fn = datasets.get_dataset("test", FLAGS)
  model_fn = models.get_model(FLAGS)
  best_iou = 0.

  with tf.summary.FileWriter(path.join(FLAGS.train_dir, "eval")) as eval_writer:
    hook_dict = build_eval_graph(input_fn, model_fn, FLAGS)
    saver = tf.train.Saver()
    for ckpt in tf.train.checkpoints_iterator(FLAGS.train_dir, timeout=1800):
      summary, global_step = evaluate(hook_dict, ckpt, saver, best_iou, FLAGS)
      eval_writer.add_summary(summary, global_step)
      eval_writer.flush()


if __name__ == "__main__":
  tf.app.run(main)
