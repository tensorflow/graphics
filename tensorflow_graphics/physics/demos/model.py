import tensorflow as tf
import numpy as np

class BatchNormalize:
  def __init__(self, shape, alpha = 0.9):
    self.mean = tf.Variable(np.zeros(shape, dtype = np.float32), trainable = False)
    self.norm = tf.Variable(np.ones(shape, dtype = np.float32), trainable = False)

    self.mean_ph = tf.compat.v1.placeholder(shape = shape, dtype = tf.float32)
    self.norm_ph = tf.compat.v1.placeholder(shape = shape, dtype = tf.float32)
    self.update_sym = [
        self.norm.assign(self.norm * alpha + self.norm_ph * (1 - alpha)),
        self.mean.assign(self.mean * alpha + self.mean_ph * (1 - alpha))
    ]
  
  def __call__(self, input_layer):
    return (input_layer - self.mean) / (self.norm ** 0.5)

  def get_mean(self):
    return self.mean

  def update(self, mean, norm, sess):
    sess.run(self.update_sym, feed_dict = {self.mean_ph: mean, self.norm_ph: norm})

class SimpleModel:
  def __init__(self, input_channel, output_channel, hidden_layer = 16, batch_normalize = False):
    if batch_normalize:
      self.bn = BatchNormalize(input_channel)
    else:
      self.bn = None
    self.W1 = tf.Variable(
        0.2 * tf.random.normal(shape=(input_channel, hidden_layer)),
        trainable=True)
    self.b1 = tf.Variable(np.random.random(hidden_layer) * 0.2, trainable=True, dtype = tf.float32)
  
    self.W2 = tf.Variable(
        0.2 * tf.random.normal(shape=(hidden_layer, output_channel)),
        trainable=True)
    self.b2 = tf.Variable(np.random.random(output_channel) * 0.2, trainable=True, dtype = tf.float32)

  def __call__(self, input_layer):
    if self.bn is not None:
      input_layer = self.bn(input_layer)
    hidden_layer = tf.tanh(tf.matmul(input_layer, self.W1) + self.b1)
    output_layer = tf.tanh(tf.matmul(hidden_layer, self.W2) + self.b2)
    return output_layer

  def update_bn(self, mean, norm, sess):
    self.bn.update(mean, norm, sess)

  def get_bn_mean(self):
    return self.bn.get_mean()
