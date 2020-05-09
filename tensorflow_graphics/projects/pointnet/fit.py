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
"""keras Model.fit loop for PointNet v1 on modelnet40."""
# pylint: disable=missing-function-docstring
import functools
import tensorflow as tf

from tensorflow_graphics.datasets.modelnet40 import ModelNet40
from tensorflow_graphics.nn.layer.pointnet import VanillaClassifier
from tensorflow_graphics.projects.pointnet import augment as augment_lib
from tensorflow_graphics.projects.pointnet import helpers

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser = helpers.ArgumentParser()
parser.add("--batch_size", 32)
parser.add("--num_epochs", 250)
parser.add("--num_points", 2048, help="subsampled (max 2048)")
parser.add("--learning_rate", 1e-3, help="initial Adam learning rate")
parser.add("--lr_decay", True, help="enable learning rate decay")
parser.add("--bn_decay", .5, help="batch norm decay momentum")
parser.add("--ev_every", 1, help="evaluation frequency (epochs)")
parser.add("--augment", True, help="use augmentations")
parser.add("--verbose", True, help="enable the progress bar")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def preprocess(points, labels, num_points, augment):
  points = points[:num_points]
  if augment:
    points = augment_lib.rotate(points)
    points = augment_lib.jitter(points)
  return points, labels


def get_datasets(num_points, batch_size, augment):
  (ds_train, ds_test), info = ModelNet40.load(as_supervised=True,
                                              split=("train", "test"),
                                              with_info=True)
  num_examples = info.splits["train"].num_examples
  ds_train = ds_train.shuffle(num_examples, reshuffle_each_iteration=True)
  ds_train = ds_train.map(
      functools.partial(preprocess, num_points=num_points, augment=augment),
      tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.map(
      functools.partial(preprocess, num_points=num_points, augment=False),
      tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  return ds_train, ds_test


def get_optimizer(learning_rate, lr_decay=None):
  if lr_decay is not None:
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=6250,  #< 200.000 / 32 (batch size) (from original pointnet)
        decay_rate=0.7,
        staircase=True)
    return tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def main():
  FLAGS = parser.parse_args()  # pylint:disable=invalid-name
  points = tf.keras.Input((FLAGS.num_points, 3), dtype=tf.float32)
  logits = VanillaClassifier(num_classes=40, momentum=FLAGS.bn_decay)(points)
  model = tf.keras.Model(points, logits)
  model.compile(
      optimizer=get_optimizer(learning_rate=FLAGS.learning_rate,
                              lr_decay=FLAGS.lr_decay),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=tf.keras.metrics.SparseCategoricalAccuracy())
  ds_train, ds_test = get_datasets(FLAGS.num_points, FLAGS.batch_size,
                                   FLAGS.augment)
  history = model.fit(ds_train,
                      epochs=FLAGS.num_epochs,
                      validation_freq=FLAGS.ev_every,
                      validation_data=ds_test,
                      callbacks=[tf.keras.callbacks.TensorBoard(FLAGS.logdir)],
                      verbose=FLAGS.verbose)
  return model, history


if __name__ == '__main__':
  main()
