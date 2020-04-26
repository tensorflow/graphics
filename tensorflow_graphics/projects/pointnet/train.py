#Copyright 2019 Google LLC
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

"""Training loop for PointNet v1 on modelnet40."""
# pylint: disable=missing-function-docstring

import tensorflow as tf
from tqdm import tqdm
import helpers
import augment

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from tensorflow_graphics.datasets.modelnet40 import ModelNet40
from tensorflow_graphics.nn.layer.pointnet import PointNetVanillaClassifier as PointNet

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = helpers.ArgumentParser()
parser.add("--batch_size", 32)
parser.add("--num_epochs", 250)
parser.add("--learning_rate", 1e-3, help="initial Adam learning rate")
parser.add("--num_points", 2048, help="subsampled (max 2048)")
parser.add("--lr_decay", True, help="use Adam learning rate decay")
parser.add("--tb_every", 100, help="tensorboard frequency (itereations)")
parser.add("--bn_decay", .5, help="batch norm decay momentum")
parser.add("--tqdm", True, help="enable the progress bar")
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

ds_train, info = ModelNet40.load(split="train", with_info=True)
ds_train = ds_train.shuffle(2048, reshuffle_each_iteration=True)  # < TODO meaning?
ds_train = ds_train.batch(FLAGS.batch_size)
ds_test = ModelNet40.load(split="test").batch(FLAGS.batch_size)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if FLAGS.lr_decay:
  lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
      FLAGS.learning_rate,
      decay_steps=6250, #< 200.000 / 32 (batch size) (from original pointnet)
      decay_rate=0.7,
      staircase=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
else:
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

num_classes = info.features["label"].num_classes
model = PointNet(num_classes=num_classes, momentum=FLAGS.bn_decay)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @tf.function
def wrapped_tf_function(points, labels):
  with tf.GradientTape() as tape:
    logits = model(points, training=True)
    loss = model.loss(labels, logits)
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss

def train_step(example):
  step = optimizer.iterations.numpy()
  points, labels = example["points"], example["label"]

  # --- subsampling and augmentation
  points = points[0:FLAGS.num_points]
  points = augment.rotate(points)
  points = augment.jitter(points)

  # --- optimize
  loss = wrapped_tf_function(points, labels)
  if step % FLAGS.tb_every == 0:
    tf.summary.scalar(name="loss", data=loss, step=step)

  # --- report rate in summaries
  if FLAGS.lr_decay and step % FLAGS.tb_every == 0:
    tf.summary.scalar(name="learning_rate", data=lr_scheduler(step), step=step)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def train_epoch(epoch):
  for example in tqdm(ds_train, leave=False, disable=not FLAGS.tqdm):
    train_step(example)
  evaluate(epoch)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def evaluate(epoch):
  if "best_accuracy" not in evaluate.__dict__:
    evaluate.best_accuracy = 0
  aggregator = tf.keras.metrics.SparseCategoricalAccuracy()
  step = optimizer.iterations.numpy()

  for example in ds_test:
    points, labels = example["points"], example["label"]
    logits = model(points, training=False)
    aggregator.update_state(labels, logits)

  accuracy = aggregator.result()
  evaluate.best_accuracy = max(accuracy, evaluate.best_accuracy)
  tf.summary.scalar(name="accuracy_test", data=accuracy, step=step)
  print("[epoch {}, step {}] accuracy: {:.3f}, best: {:.3f}".format(
    epoch, step, accuracy, evaluate.best_accuracy))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == "__main__":
  try:
    helpers.setup_tensorboard(FLAGS)
    helpers.summary_command(parser, FLAGS)
    for iepoch in range(FLAGS.num_epochs):
      train_epoch(iepoch)
  except KeyboardInterrupt:
    helpers.handle_keyboard_interrupt(FLAGS)
