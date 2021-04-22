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
import tensorflow as tf

from tensorflow_graphics.datasets import modelnet40
from tensorflow_graphics.nn.layer.pointnet import VanillaClassifier
from tensorflow_graphics.projects.pointnet import augment
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
parser.add("--augment_jitter", True, help="use jitter augmentation")
parser.add("--augment_rotate", True, help="use rotate augmentation")
parser.add("--verbose", True, help="enable the progress bar")
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if FLAGS.lr_decay:
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
      FLAGS.learning_rate,
      decay_steps=6250,  #< 200.000 / 32 (batch size) (from original pointnet)
      decay_rate=0.7,
      staircase=True)
else:
  learning_rate = FLAGS.learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

points = tf.keras.Input((FLAGS.num_points, 3), dtype=tf.float32)
logits = VanillaClassifier(num_classes=40, momentum=FLAGS.bn_decay)(points)
model = tf.keras.Model(points, logits)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

(ds_train, ds_test), info = modelnet40.ModelNet40().load(split=("train",
                                                                "test"),
                                                         with_info=True,
                                                         as_supervised=True)


# Train data pipeling
def augment_train(points, label):
  points = points[:FLAGS.num_points]
  if FLAGS.augment_jitter:
    points = augment.jitter(points, stddev=0.01, clip=0.05)
  if FLAGS.augment_rotate:
    points = augment.rotate(points)
  return points, label


num_examples = info.splits["train"].num_examples
ds_train = ds_train.shuffle(num_examples, reshuffle_each_iteration=True)
ds_train = ds_train.map(augment_train,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(FLAGS.batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# Test data pipeline
def augment_test(points, label):
  return points[:FLAGS.num_points], label


ds_test = ds_test.map(augment_test)
ds_test = ds_test.batch(FLAGS.batch_size)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.TensorBoard(FLAGS.logdir, profile_batch='2,12'),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

try:
  model.fit(ds_train,
            epochs=FLAGS.num_epochs,
            validation_freq=FLAGS.ev_every,
            validation_data=ds_test,
            callbacks=callbacks,
            verbose=FLAGS.verbose)
except KeyboardInterrupt:
  helpers.handle_keyboard_interrupt(FLAGS)
