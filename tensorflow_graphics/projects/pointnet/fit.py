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
import tensorflow as tf

from tensorflow_graphics.nn.layer.pointnet import VanillaClassifier
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
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

points = tf.keras.Input((FLAGS.num_points, 3), dtype=tf.float32)
logits = VanillaClassifier(num_classes=40, momentum=FLAGS.bn_decay)(points)
model = tf.keras.Model(points, logits)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=helpers.decayed_learning_rate(FLAGS.learning_rate) if FLAGS.
    lr_decay else FLAGS.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

ds_train, ds_test = helpers.get_modelnet40_datasets(FLAGS.num_points,
                                                    FLAGS.batch_size,
                                                    FLAGS.augment)

callbacks = [
    tf.keras.callbacks.TensorBoard(FLAGS.logdir),
    tf.keras.callbacks.ModelCheckpoint(FLAGS.logdir, save_best_only=True),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

model.fit(ds_train,
          epochs=FLAGS.num_epochs,
          validation_freq=FLAGS.ev_every,
          validation_data=ds_test,
          callbacks=callbacks,
          verbose=FLAGS.verbose)
