#Copyright 2020 Google LLC
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

"""
Implementation (incomplete) of the network from the paper:

@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}

NOTE: "batchnorm_momentum" is referred to as "bn_decay" in the original implementation. 
NOTE: scheduling of batchnorm_momentum currently not possible (due to keras)
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

class VanillaEncoder(object):
  def __init__(self, num_points, batchnorm_momentum=.5):
    # TODO check whether Conv2D is really more efficient than Dense
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(64, (1, 1), input_shape=(num_points, 1, 3)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Conv2D(64, (1, 1)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Conv2D(64, (1, 1)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Conv2D(128, (1, 1)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Conv2D(1024, (1, 1)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.MaxPool2D(pool_size=(num_points, 1)))
    self.model.add(layers.Flatten())
    self.trainable_variables = self.model.trainable_variables

  def __call__(self, x, training):
    x = tf.expand_dims(x, axis=2) #< BxNx1xD (prep for Conv2D)
    return self.model(x, training) #< Bx1024


class ClassificationHead(object):
  def __init__(self, num_classes=40, n_features=1024, batchnorm_momentum=.5):
    self.model = models.Sequential()
    self.model.add(layers.Dense(512, input_shape=(n_features,)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Dense(256))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Dropout(0.3))
    self.model.add(layers.Dense(num_classes, activation="linear"))
    self.trainable_variables = self.model.trainable_variables

  def __call__(self, features, training):
    return self.model(features, training) #< Bx1


class PointNetVanillaClassifier(object):
  
  def __init__(self, num_points, num_classes, batchnorm_momentum=.5):
    self.encoder = VanillaEncoder(num_points=num_points, batchnorm_momentum=batchnorm_momentum)
    self.classifier = ClassificationHead(num_classes=num_classes, batchnorm_momentum=batchnorm_momentum)
    self.trainable_variables = self.encoder.trainable_variables + self.classifier.trainable_variables 
  
  def __call__(self, points, training):
    features = self.encoder(points, training) #< Bx1024
    logits = self.classifier(features, training) #< Bx40
    return logits

  @staticmethod
  def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
    residual = cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(residual)
    return loss