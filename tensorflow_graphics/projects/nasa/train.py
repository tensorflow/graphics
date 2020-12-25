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


def main(unused_argv):
  tf.random.set_random_seed(20200823)
  np.random.seed(20200823)

  logging.info("=> Starting ...")

  # Select dataset.
  logging.info("=> Preparing datasets ...")
  input_fn = datasets.get_dataset("train", FLAGS)

  # Select model.
  logging.info("=> Creating {} model".format(FLAGS.model))
  model_fn = models.get_model(FLAGS)

  # Set up training.
  logging.info("=> Setting up training ...")
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.train_dir,
      save_checkpoints_steps=FLAGS.save_every,
      save_summary_steps=FLAGS.summary_every,
      keep_checkpoint_max=None,
  )
  trainer = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
  )

  # Start training.
  logging.info("=> Training ...")
  trainer.train(input_fn=input_fn, max_steps=FLAGS.max_steps)


if __name__ == "__main__":
  tf.app.run(main)
