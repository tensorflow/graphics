import tensorflow as tf
import numpy as np
from typing import Tuple
import IPython

use_float64 = False

if not use_float64:
  np_precision = np.float32
  tf_precision = tf.float32
else:
  np_precision = np.float64
  tf_precision = tf.float64

# (b, X, Y, p)
identity_matrix = np.array([[1, 0], [0, 1]], dtype=np_precision)[None, :, :, None]
identity_matrix_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np_precision)[None, :, :, None]


def make_matrix2d_from_scalar(m00 : float, m01 : float, m10 : float, m11 : float) -> tf.Tensor:
  """
  Create a 2D matrix from four scalars
  
  :return: The 2D matrix
  :rtype: tf.Tensor
  """
  m00 = tf.ones(shape=(1, 1), dtype=tf_precision) * m00
  m01 = tf.ones(shape=(1, 1), dtype=tf_precision) * m01
  m10 = tf.ones(shape=(1, 1), dtype=tf_precision) * m10
  m11 = tf.ones(shape=(1, 1), dtype=tf_precision) * m11
  return make_matrix2d(m00, m01, m10, m11)

 
def make_matrix2d(m00 : tf.Tensor, m01 : tf.Tensor, m10 : tf.Tensor, m11 : tf.Tensor) -> tf.Tensor:
  """
  Create a matrix of size batch x 2 x 2 x num_particles from four tensors of size batch_size x num_particles
  
  :return: The 2D matrix
  :rtype: tf.Tensor
  """
  assert len(m00.shape) == 2  # Batch, particles
  assert len(m01.shape) == 2  # Batch, particles
  assert len(m10.shape) == 2  # Batch, particles
  assert len(m11.shape) == 2  # Batch, particles
  row0 = tf.stack([m00, m01], axis=1)
  row1 = tf.stack([m10, m11], axis=1)
  return tf.stack([row0, row1], axis=1)


def make_matrix3d(m00 : tf.Tensor, m01 : tf.Tensor, m02 : tf.Tensor, 
                  m10 : tf.Tensor, m11 : tf.Tensor, m12 : tf.Tensor, 
                  m20 : tf.Tensor, m21 : tf.Tensor, m22 : tf.Tensor) -> tf.Tensor:
  """
  Create a matrix of size batch x 3 x 3 x num_particles from nine tensors of size batch_size x num_particles
  
  :return: The 3D matrix
  :rtype: tf.Tensor
  """
  assert len(m00.shape) == 2  # Batch, particles
  assert len(m01.shape) == 2  # Batch, particles
  assert len(m10.shape) == 2  # Batch, particles
  assert len(m11.shape) == 2  # Batch, particles
  row0 = tf.stack([m00, m01, m02], axis=1)  
  row1 = tf.stack([m10, m11, m12], axis=1)
  row2 = tf.stack([m20, m21, m22], axis=1)
  return tf.stack([row0, row1, row2], axis=1)

def polar_decomposition(m: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  Reference: http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
  Return the 2D Polar Decomposition of the input matrix m.
  
  :param m: Matrix of batch_size x 2 x 2 x num_particles
  :type m: tf.Tensor
  :return r: Rotation Matrix of size batch x 2 x 2 x num_particles
  :rtype r: tf.Tensor
  :return s: Scaling Matrix of size batch x 2 x 2 x num_particles
  :rtype s: tf.Tensor
  """
  # 
  assert len(m.shape) == 4  # Batch, row, column, particles
  x = m[:, 0, 0, :] + m[:, 1, 1, :]
  y = m[:, 1, 0, :] - m[:, 0, 1, :]
  scale = 1.0 / tf.sqrt(x**2 + y**2)
  c = x * scale
  s = y * scale
  r = make_matrix2d(c, -s, s, c)
  return r, matmatmul(transpose(r), m)


def inverse(m : tf.Tensor) -> tf.Tensor:
  """
  Reference: http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
  Return the 2D Inverse of the input matrix m.
  
  :param m: Matrix of batch_size x 2 x 2 x num_particles
  :type m: tf.Tensor
  :return: Inverse Matrix of size batch x 2 x 2 x num_particles
  :rtype: tf.Tensor
  """
  assert len(m.shape) == 4  # Batch, row, column, particles
  Jinv = 1.0 / determinant(m)
  return Jinv[:, None, None, :] * make_matrix2d(m[:, 1, 1, :], -m[:, 0, 1, :],
                                                -m[:, 1, 0, :], m[:, 0, 0, :])


def matmatmul(a : tf.Tensor, b : tf.Tensor) -> tf.Tensor:
  """
  Perform the matrix multiplication of two 2-D input matrices a @ b
  
  :param a: Matrix of batch_size x 2 x 2 x num_particles
  :type a: tf.Tensor
  :param b: Matrix of batch_size x 2 x 2 x num_particles
  :type b: tf.Tensor
  :return: Matrix multiplication of size batch x 2 x 2 x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 4  # Batch, row, column, particles
  assert len(b.shape) == 4  # Batch, row, column, particles
  dim = a.shape[1]
  assert a.shape[2] == b.shape[1]
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      for k in range(dim):
        if k == 0:
          c[i][j] = a[:, i, k, :] * b[:, k, j, :]
        else:
          c[i][j] += a[:, i, k, :] * b[:, k, j, :]
  if dim == 2:
    return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])
  else:
    return make_matrix3d(c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2])

def at_b(a, b):
  assert len(a.shape) == 4  # Batch, row, column, particles
  assert len(b.shape) == 4
  assert a.shape[1] == a.shape[2]
  assert b.shape[1] == b.shape[2]
  assert a.shape[1] == b.shape[1]
  dim = a.shape[1]
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      for k in range(dim):
        if k == 0:
          c[i][j] = a[:, k, i, :] * b[:, k, j, :]
        else:
          c[i][j] += a[:, k, i, :] * b[:, k, j, :]
  if dim == 2:
    return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])
  else:
    return make_matrix3d(c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2])

def Mt_M(a : tf.Tensor) -> tf.Tensor:
  """
  Perform the operation matmul(transpose(a) , a) for an input matrix a
  
  :param a: Matrix of batch_size x dim x dim x num_particles
  :type m: tf.Tensor
  :return: matmul(transpose(a) , a) of size batch x dim x dim x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 4  # Batch, row, column, particles
  assert a.shape[1] == a.shape[2]
  dim = a.shape[1]
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      for k in range(dim):
        if k == 0:
          c[i][j] = a[:, k, i, :] * a[:, k, j, :]
        else:
          c[i][j] += a[:, k, i, :] * a[:, k, j, :]
  if dim == 2:
    return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])
  else:
    return make_matrix3d(c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2])


def transpose(a : tf.Tensor) -> tf.Tensor:
  """
  Compute transpose(a)
  
  :param a: Matrix of batch_size x dim x dim x num_particles
  :type a: tf.Tensor
  :return: transpose(a) of batch_sixe x dim x dim x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 4  # Batch, row, column, particles
  dim = a.shape[1]
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      c[i][j] = a[:, j, i, :]
  #c[0][1], c[1][0] = c[1][0], c[0][1]
  #row0 = tf.stack([c[0][0], c[0][1]], axis=1)
  #row1 = tf.stack([c[1][0], c[1][1]], axis=1)
  #C = tf.stack([row0, row1], axis=1)
  if dim == 2:
    return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])
  else:
    return make_matrix3d(c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2])
  
def M_plus_Mt(a):
  assert len(a.shape) == 4  # Batch, row, column, particles
  assert a.shape[1] == a.shape[2]
  dim = a.shape[1]
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      c[i][j] = a[:, i, j, :] + a[:, j, i, :]
  if dim == 2:
    return make_matrix2d(c[0][0], c[0][1], c[1][0], c[1][1])
  else:
    return make_matrix3d(c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2])


def matvecmul(a : tf.Tensor, b : tf.Tensor) -> tf.Tensor:
  """
  Compute matrix-vector product for matrix a and vector b
  
  :param a: Matrix of batch_size x row x col x num_particles
  :type a: tf.Tensor
  :param b: Vector of batch_size x col x num_particles
  :type b: tf.Tensor
  :return: a * b of batch_sixe x row x num_particles
  :rtype: tf.Tensor
  """

  assert len(a.shape) == 4  # Batch, row, column, particles
  assert len(b.shape) == 3  # Batch, column, particles
  row, col = a.shape[1:3]
  c = [None for i in range(row)]
  for i in range(row):
    for k in range(col):
      if k == 0:
        c[i] = a[:, i, k, :] * b[:, k, :]
      else:
        c[i] += a[:, i, k, :] * b[:, k, :]
  return tf.stack(c, axis=1)


def matvecmul_grid(a, b):
  """
  Compute matrix-vector product for matrix a and vector b.  Essentially rank 3 batching.
  
  :param a: Matrix of batch_size x grid_x x grid_y row x col
  :type a: tf.Tensor
  :param b: Vector of batch_size x grid_x x grid_y x col
  :type b: tf.Tensor
  :return: a * b of batch_size x grid_x x grid_y x row
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 5 # Batch, grid_x, grid_y, row, column
  assert len(b.shape) == 4 # Batch, grid_x, grid_y, column
  row, col = a.shape[3:]
  c = [None for i in range(row)]
  for i in range(row):
    for k in range(col):
      if k == 0:
        c[i] = a[:, :, :, i, k] * b[:, :, :, k]
      else:
        c[i] += a[:, :, :, i, k] * b[:, :, :, k]
  return tf.stack(c, axis=3)

# a is column, b is row
def outer_product(a : tf.Tensor, b : tf.Tensor) -> tf.Tensor:
  """
  Compute outer product of column vector a and row vector b.
  
  :param a: vector of batch_size x dim x num_particles
  :type a: tf.Tensor
  :param b: vector of batch_size x dim x num_particles
  :type b: tf.Tensor
  :return: outer product of size batch_size x dim x dim x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 3  # Batch, row, particles
  assert len(b.shape) == 3  # Batch, row, particles
  dim = 2
  c = [[None for i in range(dim)] for j in range(dim)]
  for i in range(dim):
    for j in range(dim):
      c[i][j] = a[:, i, :] * b[:, j, :]
  row0 = tf.stack([c[0][0], c[0][1]], axis=1)
  row1 = tf.stack([c[1][0], c[1][1]], axis=1)
  C = tf.stack([row0, row1], axis=1)
  return C


def determinant(a : tf.Tensor) -> tf.Tensor:
  """
  Compute determinant of a 2-D matrix a.
  
  :param a: vector of batch_size x 2 x 2 x num_particles
  :type a: tf.Tensor
  :return: determinant of size batch_size x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 4  # Batch, row, column, particles
  return a[:, 0, 0, :] * a[:, 1, 1, :] - a[:, 1, 0, :] * a[:, 0, 1, :]


# returns (b, p)
def trace(a : tf.Tensor) -> tf.Tensor:
  """
  Compute trace of a 2-D matrix a.
  
  :param a: vector of batch_size x 2 x 2 x num_particles
  :type a: tf.Tensor
  :return: trace of size batch_size x num_particles
  :rtype: tf.Tensor
  """
  assert len(a.shape) == 4  # Batch, row, column, particles
  return a[:, 0, 0, :] + a[:, 1, 1, :]


if __name__ == '__main__':
  a = np.random.randn(2, 2)
  b = np.random.randn(2, 2)
  c = np.random.randn(2, 1)
  d = np.random.randn(2, 1)
  with tf.Session() as sess:
    # Polar decomposition
    R, S = polar_decomposition(tf.constant(a[None, :, :, None]))
    r, s = sess.run([R, S])
    r = r[0, :, :, 0]
    s = s[0, :, :, 0]
    np.testing.assert_array_almost_equal(np.matmul(r, s), a)
    np.testing.assert_array_almost_equal(
        np.matmul(r, np.transpose(r)), [[1, 0], [0, 1]])
    np.testing.assert_array_almost_equal(s, np.transpose(s))

    # Inverse
    prod2 = inverse(tf.constant(a[None, :, :, None]))
    prod2 = sess.run(prod2)[0, :, :, 0]
    np.testing.assert_array_almost_equal(np.matmul(prod2, a), [[1, 0], [0, 1]])

    # Matmatmul
    prod1 = np.matmul(a, b)
    prod2 = matmatmul(
        tf.constant(a[None, :, :, None]), tf.constant(b[None, :, :, None]))
    prod2 = sess.run(prod2)[0, :, :, 0]
    np.testing.assert_array_almost_equal(prod1, prod2)

    # Matvecmul
    prod1 = np.matmul(a, c)
    prod2 = matvecmul(
        tf.constant(a[None, :, :, None]), tf.constant(c[None, :, 0, None]))
    prod2 = sess.run(prod2)[0, :, 0]
    np.testing.assert_array_almost_equal(prod1[:, 0], prod2)

    # Matvecmul on grid.
    batch_size = 1
    nx, ny = 10, 6
    dx, dy = 3, 2
    grid_R = np.random.rand(batch_size, nx, ny, dx, dy)
    grid_vel = np.random.rand(batch_size, nx, ny, dy)
    grid_result1 = matvecmul_grid(tf.constant(grid_R), tf.constant(grid_vel))
    grid_result1 = sess.run(grid_result1)
    assert grid_result1.shape == (batch_size, nx, ny, dx)
    grid_result2 = np.zeros((batch_size, nx, ny, dx))
    for i in range(batch_size):
      for j in range(nx):
        for k in range(ny):
          grid_result2[i, j, k, :] = np.matmul(grid_R[i, j, k, :, :], grid_vel[i, j, k, :])
    np.testing.assert_array_almost_equal(grid_result1, grid_result2)

    # transpose
    prod1 = np.transpose(a)
    prod2 = transpose(tf.constant(a[None, :, :, None]))
    prod2 = sess.run(prod2)[0, :, :, 0]
    np.testing.assert_array_almost_equal(prod1, prod2)

    # outer_product
    prod2 = outer_product(
        tf.constant(c[None, :, 0, None]), tf.constant(d[None, :, 0, None]))
    prod2 = sess.run(prod2)[0, :, :, 0]
    for i in range(2):
      for j in range(2):
        np.testing.assert_array_almost_equal(c[i] * d[j], prod2[i, j])
  print("All tests passed.")
