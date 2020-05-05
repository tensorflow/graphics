"Testing of pointnet module."
import sys
import pytest
import tensorflow as tf

def make_fake_batch(B, N): # pylint: disable=invalid-name
  points = tf.random.normal((B, N, 3))
  label = tf.random.uniform((B,), minval=0, maxval=40, dtype=tf.int32)
  return {"points": points, "label": label}

def test_dryrun():
  sys.argv = ["train.py", "--dryrun", "--assert_gpu", "False"]
  import train  # pylint: disable=import-outside-toplevel, unused-import
  for i in range(2):
    batch = make_fake_batch(32, 1024)
    train.train(batch)

if __name__ == "__main__":
  sys.exit(pytest.main([__file__]))
