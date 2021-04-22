import os
import time
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
from IPython import embed
import mpm3d
from simulation import Simulation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = tf.compat.v1.Session()

# Memory bandwidth of 1070 = 256 GB/s

def main():
  #shape = (1024, 1024, 256)
  shape = (1024, 256, 256) # size = 0.256 GB, walkthrough time = 1ms
  ones = np.ones(shape=shape)

  a = tf.Variable(ones, dtype=tf.float32)
  b = tf.Variable(ones, dtype=tf.float32)

  sess.run(tf.compat.v1.global_variables_initializer())
  # op = tf.reduce_max(tf.assign(a, a + b))
  # op = tf.assign(a, b)
  op = tf.reduce_max(input_tensor=a + b)
  #op = tf.assign(a, a)

  total_time = 0
  N = 1000
  for i in range(N):
    t = time.time()
    ret = sess.run(op)
    t = time.time() - t
    print(t)
    total_time += t
  print(total_time / N * 1000, 'ms')

if __name__ == '__main__':
  main()


# Reduce max a (1)    : 1.61 ms
# Reduce max a + b (4): 5.57 ms
# a=b, fetch          : 23 ms
