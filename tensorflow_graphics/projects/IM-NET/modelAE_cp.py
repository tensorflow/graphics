import tensorflow as tf
import tensorflow_addons as tfa

import os
import time
import math
import random
import numpy as np
import h5py

import mcubes

from utils import *
from utils import leaky_relu

from tqdm import tqdm


class generator(tf.keras.Model):
  def __init__(self, z_dim, point_dim, gf_dim):
    super(generator, self).__init__()
    self.z_dim = z_dim
    self.point_dim = point_dim
    self.gf_dim = gf_dim
    self.linear_1 = tf.keras.layers.Dense(
        self.gf_dim * 8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_2 = tf.keras.layers.Dense(
        self.gf_dim * 8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_3 = tf.keras.layers.Dense(
        self.gf_dim * 8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_4 = tf.keras.layers.Dense(
        self.gf_dim * 4, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_5 = tf.keras.layers.Dense(
        self.gf_dim * 2, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_6 = tf.keras.layers.Dense(
        self.gf_dim * 1, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_7 = tf.keras.layers.Dense(
        1, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(1e-5, 0.02), bias_initializer='zeros')

  def __call__(self, points, z, training=False):
    zs = tf.broadcast_to(tf.reshape(
        z, [-1, 1, self.z_dim]), [z.shape[0], points.shape[1], self.z_dim])
    pointz = tf.concat([points, zs], axis=2)

    l1 = self.linear_1(pointz)
    l1 = leaky_relu(l1, 0.02)

    l2 = self.linear_2(l1)
    l2 = leaky_relu(l2, 0.02)

    l3 = self.linear_3(l2)
    l3 = leaky_relu(l3, 0.02)

    l4 = self.linear_4(l3)
    l4 = leaky_relu(l4, 0.02)

    l5 = self.linear_5(l4)
    l5 = leaky_relu(l5, 0.02)

    l6 = self.linear_6(l5)
    l6 = leaky_relu(l6, 0.02)

    l7 = self.linear_7(l6)

    l7 = tf.math.maximum(tf.math.minimum(l7, l7*0.01+0.99), l7*0.01)

    return l7


class encoder(tf.keras.Model):
  def __init__(self, ef_dim, z_dim):
    super(encoder, self).__init__()
    self.ef_dim = ef_dim
    self.z_dim = z_dim
    self.conv_1 = tf.keras.layers.Conv3D(
        self.ef_dim, 4, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.in_1 = tfa.layers.InstanceNormalization()
    self.conv_2 = tf.keras.layers.Conv3D(
        self.ef_dim * 2, 4, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.in_2 = tfa.layers.InstanceNormalization()
    self.conv_3 = tf.keras.layers.Conv3D(
        self.ef_dim * 4, 4, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.in_3 = tfa.layers.InstanceNormalization()
    self.conv_4 = tf.keras.layers.Conv3D(
        self.ef_dim * 8, 4, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.in_4 = tfa.layers.InstanceNormalization()
    self.conv_5 = tf.keras.layers.Conv3D(
        self.z_dim, 4, strides=2, padding='valid', use_bias=True, kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer='zeros')

  def __call__(self, inputs, training=False):
    d_1 = self.in_1(self.conv_1(inputs), training=training)
    d_1 = leaky_relu(d_1, 0.02)

    d_2 = self.in_2(self.conv_2(d_1), training=training)
    d_2 = leaky_relu(d_2, 0.02)

    d_3 = self.in_3(self.conv_3(d_2), training=training)
    d_3 = leaky_relu(d_3, 0.02)

    d_4 = self.in_4(self.conv_4(d_3), training=training)
    d_4 = leaky_relu(d_4, 0.02)

    d_5 = self.conv_5(d_4)
    d_5 = tf.reshape(d_5, [-1, self.z_dim])

    d_5 = tf.keras.activations.sigmoid(d_5)

    return d_5


class im_network(tf.keras.Model):
  def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
    super(im_network, self).__init__()
    self.ef_dim = ef_dim
    self.gf_dim = gf_dim
    self.z_dim = z_dim
    self.point_dim = point_dim
    self.encoder = encoder(self.ef_dim, self.z_dim)
    self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)

  def __call__(self, inputs, z_vector, point_coord, training=False):
    if training:
      z_vector = self.encoder(inputs, training=training)
      net_out = self.generator(
          point_coord, z_vector, training=training)
    else:
      if inputs is not None:
        z_vector = self.encoder(inputs, training=training)
      if z_vector is not None and point_coord is not None:
        net_out = self.generator(
            point_coord, z_vector, training=training)
      else:
        net_out = None

    return z_vector, net_out


class IM_AE(object):
  def __init__(self, config):
    # progressive training
    # 1-- (16, 16*16*16)
    # 2-- (32, 16*16*16)
    # 3-- (64, 16*16*16*4)
    self.sample_vox_size = config.sample_vox_size
    if self.sample_vox_size == 16:
      self.load_point_batch_size = 16*16*16
      self.point_batch_size = 16*16*16
      self.shape_batch_size = 32
    elif self.sample_vox_size == 32:
      self.load_point_batch_size = 16*16*16
      self.point_batch_size = 16*16*16
      self.shape_batch_size = 32
    elif self.sample_vox_size == 64:
      self.load_point_batch_size = 16*16*16*4
      self.point_batch_size = 16*16*16
      self.shape_batch_size = 32
    self.input_size = 64  # input voxel grid size

    self.ef_dim = 32
    self.gf_dim = 128
    self.z_dim = 256
    self.point_dim = 3

    self.dataset_name = config.dataset
    self.dataset_load = self.dataset_name + '_train'
    if not (config.train or config.getz):
      self.dataset_load = self.dataset_name + '_test'
    self.checkpoint_dir = config.checkpoint_dir
    self.data_dir = config.data_dir

    data_hdf5_name = self.data_dir + '/'+self.dataset_load + '.hdf5'
    if os.path.exists(data_hdf5_name):
      data_dict = h5py.File(data_hdf5_name, 'r')
      self.data_points = (
          data_dict['points_' + str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
      self.data_values = data_dict['values_' +
                                   str(self.sample_vox_size)][:].astype(np.float32)
      self.data_voxels = data_dict['voxels'][:]
      # reshape to NCHW
      self.data_voxels = np.reshape(
          self.data_voxels, [-1, 1, self.input_size, self.input_size, self.input_size])
    else:
      print("error: cannot load "+data_hdf5_name)
      exit(0)

    # build model
    self.im_network = im_network(
        self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)

    # TODO:
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=0.999)

    self.ckpt = tf.train.Checkpoint(
        model=self.im_network, optimizer=self.optimizer)

    # TODO:
    self.max_to_keep = 2
    self.checkpoint_path = os.path.join(
        self.checkpoint_dir, self.model_dir)
    self.checkpoint_name = 'IM_AE.model'
    self.checkpoint_manager_list = [None] * self.max_to_keep
    self.checkpoint_manager_pointer = 0

    # loss
    def network_loss(G, point_value):
      return tf.reduce_mean((G-point_value)**2)
    self.loss = network_loss

    # keep everything a power of 2
    self.cell_grid_size = 4
    self.frame_grid_size = 64
    # =256, output point-value voxel grid size in testing
    self.real_size = self.cell_grid_size*self.frame_grid_size
    self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
    self.test_point_batch_size = self.test_size * \
        self.test_size*self.test_size  # do not change

    # get coords for training
    dima = self.test_size
    dim = self.frame_grid_size
    self.aux_x = np.zeros([dima, dima, dima], np.uint8)
    self.aux_y = np.zeros([dima, dima, dima], np.uint8)
    self.aux_z = np.zeros([dima, dima, dima], np.uint8)
    multiplier = int(dim/dima)
    multiplier2 = multiplier*multiplier
    multiplier3 = multiplier*multiplier*multiplier
    for i in range(dima):
      for j in range(dima):
        for k in range(dima):
          self.aux_x[i, j, k] = i*multiplier
          self.aux_y[i, j, k] = j*multiplier
          self.aux_z[i, j, k] = k*multiplier
    self.coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
    for i in range(multiplier):
      for j in range(multiplier):
        for k in range(multiplier):
          self.coords[i*multiplier2+j*multiplier +
                      k, :, :, :, 0] = self.aux_x+i
          self.coords[i*multiplier2+j*multiplier +
                      k, :, :, :, 1] = self.aux_y+j
          self.coords[i*multiplier2+j*multiplier +
                      k, :, :, :, 2] = self.aux_z+k
    self.coords = (self.coords.astype(np.float32)+0.5)/dim-0.5
    self.coords = np.reshape(
        self.coords, [multiplier3, self.test_point_batch_size, 3])
    self.coords = tf.convert_to_tensor(self.coords)

    # get coords for testing
    dimc = self.cell_grid_size
    dimf = self.frame_grid_size
    self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
    self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
    self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
    self.cell_coords = np.zeros(
        [dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
    self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
    self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
    self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
    self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
    for i in range(dimc):
      for j in range(dimc):
        for k in range(dimc):
          self.cell_x[i, j, k] = i
          self.cell_y[i, j, k] = j
          self.cell_z[i, j, k] = k
    for i in range(dimf):
      for j in range(dimf):
        for k in range(dimf):
          self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x+i*dimc
          self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y+j*dimc
          self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z+k*dimc
          self.frame_coords[i, j, k, 0] = i
          self.frame_coords[i, j, k, 1] = j
          self.frame_coords[i, j, k, 2] = k
          self.frame_x[i, j, k] = i
          self.frame_y[i, j, k] = j
          self.frame_z[i, j, k] = k
    self.cell_coords = (self.cell_coords.astype(
        np.float32)+0.5)/self.real_size-0.5
    self.cell_coords = np.reshape(
        self.cell_coords, [dimf, dimf, dimf, dimc*dimc*dimc, 3])
    self.cell_x = np.reshape(self.cell_x, [dimc*dimc*dimc])
    self.cell_y = np.reshape(self.cell_y, [dimc*dimc*dimc])
    self.cell_z = np.reshape(self.cell_z, [dimc*dimc*dimc])
    self.frame_x = np.reshape(self.frame_x, [dimf*dimf*dimf])
    self.frame_y = np.reshape(self.frame_y, [dimf*dimf*dimf])
    self.frame_z = np.reshape(self.frame_z, [dimf*dimf*dimf])
    self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/dimf-0.5
    self.frame_coords = np.reshape(self.frame_coords, [dimf*dimf*dimf, 3])

    self.sampling_threshold = 0.5  # final marching cubes threshold

  @property
  def model_dir(self):
    return "{}_ae_{}".format(self.dataset_name, self.input_size)

  def train(self, config):
    # TODO:
    # load previous checkpoint
    if os.path.exists(self.checkpoint_path):
      print("#######################")
      print("tf.train.get_checkpoint_state{}".format(
          tf.train.get_checkpoint_state(self.checkpoint_path, latest_filename=None)))
      self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))
      print(tf.train.latest_checkpoint(self.checkpoint_path))
      print("#######################")
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    shape_num = len(self.data_voxels)
    batch_index_list = np.arange(shape_num)

    print("\n\n----------net summary----------")
    print("training samples   ", shape_num)
    print("-------------------------------\n\n")

    start_time = time.time()
    assert config.epoch == 0 or config.iteration == 0
    training_epoch = config.epoch + int(config.iteration/shape_num)
    batch_num = int(shape_num/self.shape_batch_size)
    point_batch_num = int(self.load_point_batch_size/self.point_batch_size)

    for epoch in range(0, training_epoch):
      # training
      np.random.shuffle(batch_index_list)
      avg_loss_sp = 0
      avg_num = 10
      for idx in tqdm(range(batch_num)):
        dxb = batch_index_list[idx *
                               self.shape_batch_size:(idx+1)*self.shape_batch_size]
        batch_voxels = self.data_voxels[dxb].astype(np.float32)
        if point_batch_num == 1:
          point_coord = self.data_points[dxb]
          point_value = self.data_values[dxb]
        else:
          which_batch = np.random.randint(point_batch_num)
          point_coord = self.data_points[dxb, which_batch*self.point_batch_size:(
              which_batch+1)*self.point_batch_size]
          point_value = self.data_values[dxb, which_batch*self.point_batch_size:(
              which_batch+1)*self.point_batch_size]

        batch_voxels = batch_voxels.transpose(0, 2, 3, 4, 1)

        batch_voxel = tf.convert_to_tensor(batch_voxels)
        point_coord = tf.convert_to_tensor(point_coord)
        point_value = tf.convert_to_tensor(point_value)

        # TODO:
        with tf.GradientTape() as tape:
          _, net_out = self.im_network(
              batch_voxels, None, point_coord, training=True)
          errSP = self.loss(net_out, point_value)
        grad_im_network = tape.gradient(
            errSP, self.im_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grad_im_network, self.im_network.trainable_weights))

        avg_loss_sp += errSP
        avg_num += 1
      print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (
          epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num))
      if epoch % 10 == 9:
        self.test_1(config, "train_" +
                    str(self.sample_vox_size)+"_"+str(epoch))
      if epoch % 20 == 19:
        if not os.path.exists(self.checkpoint_path):
          os.makedirs(self.checkpoint_path)
        # TODO:
        save_dir = os.path.join(
            self.checkpoint_path, self.checkpoint_name+str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")
        save_dir_encoder = os.path.join(
            self.checkpoint_path, self.checkpoint_name+"encoder"+str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")
        save_dir_encoder = os.path.join(
            self.checkpoint_path, self.checkpoint_name+"encoder"+str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")

        self.checkpoint_manager_pointer = (
            self.checkpoint_manager_pointer+1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
          if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
            os.remove(
                self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        # TODO:
        self.ckpt.save(save_dir)

        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(
            self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
          pointer = (self.checkpoint_manager_pointer +
                     self.max_to_keep-i) % self.max_to_keep
          if self.checkpoint_manager_list[pointer] is not None:
            fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

    if not os.path.exists(self.checkpoint_path):
      os.makedirs(self.checkpoint_path)
    # TODO:
    epoch = 400
    save_dir = os.path.join(self.checkpoint_path, self.checkpoint_name +
                            str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")
    save_dir_encoder = os.path.join(
        self.checkpoint_path, self.checkpoint_name+"_encoder"+str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")
    save_dir_generator = os.path.join(
        self.checkpoint_path, self.checkpoint_name+"_generator"+str(self.sample_vox_size)+"-"+str(epoch)+".ckpt")

    self.checkpoint_manager_pointer = (
        self.checkpoint_manager_pointer+1) % self.max_to_keep
    # delete checkpoint
    if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
      if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
        os.remove(
            self.checkpoint_manager_list[self.checkpoint_manager_pointer])
    # save checkpoint
    # TODO:

    self.im_network.encoder.save_weights(save_dir_encoder)
    self.im_network.generator.save_weights(save_dir_generator)

    # update checkpoint manager
    self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
    # write file
    checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
    fout = open(checkpoint_txt, 'w')
    for i in range(self.max_to_keep):
      pointer = (self.checkpoint_manager_pointer +
                 self.max_to_keep-i) % self.max_to_keep
      if self.checkpoint_manager_list[pointer] is not None:
        fout.write(self.checkpoint_manager_list[pointer]+"\n")
    fout.close()

  def test_1(self, config, name):
    multiplier = int(self.frame_grid_size/self.test_size)
    multiplier2 = multiplier*multiplier
    t = np.random.randint(len(self.data_voxels))
    model_float = np.zeros(
        [self.frame_grid_size+2, self.frame_grid_size+2, self.frame_grid_size+2], np.float32)
    batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
    batch_voxels = batch_voxels.transpose(0, 2, 3, 4, 1)
    batch_voxels = tf.convert_to_tensor(batch_voxels)
    z_vector, _ = self.im_network(
        batch_voxels, None, None, training=False)
    for i in range(multiplier):
      for j in range(multiplier):
        for k in range(multiplier):
          minib = i*multiplier2+j*multiplier+k
          point_coord = self.coords[minib:minib+1]
          _, net_out = self.im_network(
              None, z_vector, point_coord, training=False)
          model_float[self.aux_x+i+1, self.aux_y+j+1, self.aux_z+k+1] = np.reshape(
              net_out.numpy(), [self.test_size, self.test_size, self.test_size])

    vertices, triangles = mcubes.marching_cubes(
        model_float, self.sampling_threshold)
    vertices = (vertices.astype(np.float32)-0.5)/self.frame_grid_size-0.5
    # output ply sum
    write_ply_triangle(config.sample_dir+"/"+name +
                       ".ply", vertices, triangles)
    print("[sample]")

  def z2voxel(self, z):
    model_float = np.zeros(
        [self.real_size+2, self.real_size+2, self.real_size+2], np.float32)
    dimc = self.cell_grid_size
    dimf = self.frame_grid_size

    frame_flag = np.zeros([dimf+2, dimf+2, dimf+2], np.uint8)
    queue = []

    frame_batch_num = int(dimf**3/self.test_point_batch_size)
    assert frame_batch_num > 0

    # get frame grid values
    for i in range(frame_batch_num):
      point_coord = self.frame_coords[i*self.test_point_batch_size:(
          i+1)*self.test_point_batch_size]
      point_coord = np.expand_dims(point_coord, axis=0)
      point_coord = tf.convert_to_tensor(point_coord)
      _, model_out_ = self.im_network(
          None, z, point_coord, training=False)
      model_out = model_out_.numpy()[0]
      x_coords = self.frame_x[i *
                              self.test_point_batch_size:(i+1)*self.test_point_batch_size]
      y_coords = self.frame_y[i *
                              self.test_point_batch_size:(i+1)*self.test_point_batch_size]
      z_coords = self.frame_z[i *
                              self.test_point_batch_size:(i+1)*self.test_point_batch_size]
      frame_flag[x_coords+1, y_coords+1, z_coords+1] = np.reshape(
          (model_out > self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

    # get queue and fill up ones
    for i in range(1, dimf+1):
      for j in range(1, dimf+1):
        for k in range(1, dimf+1):
          maxv = np.max(frame_flag[i-1:i+2, j-1:j+2, k-1:k+2])
          minv = np.min(frame_flag[i-1:i+2, j-1:j+2, k-1:k+2])
          if maxv != minv:
            queue.append((i, j, k))
          elif maxv == 1:
            x_coords = self.cell_x+(i-1)*dimc
            y_coords = self.cell_y+(j-1)*dimc
            z_coords = self.cell_z+(k-1)*dimc
            model_float[x_coords+1, y_coords+1, z_coords+1] = 1.0

    print("running queue:", len(queue))
    cell_batch_size = dimc**3
    cell_batch_num = int(self.test_point_batch_size/cell_batch_size)
    assert cell_batch_num > 0
    # run queue
    while len(queue) > 0:
      batch_num = min(len(queue), cell_batch_num)
      point_list = []
      cell_coords = []
      for i in range(batch_num):
        point = queue.pop(0)
        point_list.append(point)
        cell_coords.append(
            self.cell_coords[point[0]-1, point[1]-1, point[2]-1])
      cell_coords = np.concatenate(cell_coords, axis=0)
      cell_coords = np.expand_dims(cell_coords, axis=0)
      cell_coords = tf.convert_to_tensor(cell_coords)
      _, model_out_batch_ = self.im_network(
          None, z, cell_coords, training=False)
      model_out_batch = model_out_batch_.numpy()[0]
      for i in range(batch_num):
        point = point_list[i]
        model_out = model_out_batch[i *
                                    cell_batch_size:(i+1)*cell_batch_size, 0]
        x_coords = self.cell_x+(point[0]-1)*dimc
        y_coords = self.cell_y+(point[1]-1)*dimc
        z_coords = self.cell_z+(point[2]-1)*dimc
        model_float[x_coords+1, y_coords+1, z_coords+1] = model_out

        if np.max(model_out) > self.sampling_threshold:
          for i in range(-1, 2):
            pi = point[0]+i
            if pi <= 0 or pi > dimf:
              continue
            for j in range(-1, 2):
              pj = point[1]+j
              if pj <= 0 or pj > dimf:
                continue
              for k in range(-1, 2):
                pk = point[2]+k
                if pk <= 0 or pk > dimf:
                  continue
                if (frame_flag[pi, pj, pk] == 0):
                  frame_flag[pi, pj, pk] = 1
                  queue.append((pi, pj, pk))
    return model_float

  # may introduce foldovers
  def optimize_mesh(self, vertices, z, iteration=3):
    new_vertices = np.copy(vertices)

    new_vertices_ = np.expand_dims(new_vertices, axis=0)
    new_vertices_ = tf.convert_to_tensor(new_vertices_)
    _, new_v_out_ = self.im_network(
        None, z, new_vertices_, training=False)
    new_v_out = new_v_out_.numpy()[0]

    for iter in range(iteration):
      for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
          for k in [-1, 0, 1]:
            if i == 0 and j == 0 and k == 0:
              continue
            offset = np.array(
                [[i, j, k]], np.float32)/(self.real_size*6*2**iter)
            current_vertices = vertices+offset
            current_vertices_ = np.expand_dims(
                current_vertices, axis=0)
            current_vertices_ = tf.convert_to_tensor(
                current_vertices_)
            _, current_v_out_ = self.im_network(
                None, z, current_vertices_, training=False)
            current_v_out = current_v_out_.numpy()[
                0]
            keep_flag = abs(
                current_v_out-self.sampling_threshold) < abs(new_v_out-self.sampling_threshold)
            keep_flag = keep_flag.astype(np.float32)
            new_vertices = current_vertices * \
                keep_flag+new_vertices*(1-keep_flag)
            new_v_out = current_v_out * \
                keep_flag+new_v_out*(1-keep_flag)
      vertices = new_vertices

    return vertices

  # output shape as ply

  def test_mesh(self, config):
    # load previous checkpoint
    # checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
    if os.path.exists(self.checkpoint_path):
      self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      return

    for t in range(config.start, min(len(self.data_voxels), config.end)):
      batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
      batch_voxels = batch_voxels.transpose(0, 2, 3, 4, 1)
      batch_voxels = tf.convert_to_tensor(batch_voxels_)
      model_z, _ = self.im_network(
          batch_voxels, None, None, training=False)
      model_float = self.z2voxel(model_z)

      vertices, triangles = mcubes.marching_cubes(
          model_float, self.sampling_threshold)
      vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
      # vertices = self.optimize_mesh(vertices,model_z)
      write_ply_triangle(config.sample_dir+"/"+str(t) +
                         "_vox.ply", vertices, triangles)

      print("[sample]")

  # output shape as ply and point cloud as ply

  def test_mesh_point(self, config):
    # load previous checkpoint
    # checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
    if os.path.exists(self.checkpoint_path):
      self.ckpt.restore(model_dir)
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      return

    for t in range(config.start, min(len(self.data_voxels), config.end)):
      batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
      batch_voxels = batch_voxels.transpose(0, 2, 3, 4, 1)
      batch_voxels = tf.convert_to_tensor(batch_voxels_)
      model_z, _ = self.im_network(
          batch_voxels, None, None, training=False)
      model_float = self.z2voxel(model_z)

      vertices, triangles = mcubes.marching_cubes(
          model_float, self.sampling_threshold)
      vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
      # vertices = self.optimize_mesh(vertices,model_z)
      write_ply_triangle(config.sample_dir+"/"+str(t) +
                         "_vox.ply", vertices, triangles)

      print("[sample]")

      # sample surface points
      sampled_points_normals = sample_points_triangle(
          vertices, triangles, 4096)
      np.random.shuffle(sampled_points_normals)
      write_ply_point_normal(
          config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals)

      print("[sample]")

  def get_z(self, config):
    # load previous checkpoint
    # checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
    if os.path.exists(self.checkpoint_path):
      self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      return

    hdf5_path = self.checkpoint_dir+'/'+self.model_dir + \
        '/'+self.dataset_name+'_train_z.hdf5'
    shape_num = len(self.data_voxels)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("zs", [shape_num, self.z_dim], np.float32)

    print(shape_num)
    for t in range(shape_num):
      batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
      batch_voxels = batch_voxels.transpose(0, 2, 3, 4, 1)
      batch_voxels = tf.convert_to_tensor(batch_voxels)
      out_z, _ = self.im_network(
          batch_voxels, None, None, training=False)
      hdf5_file["zs"][t:t+1, :] = out_z.numpy()

    hdf5_file.close()
    print("[z]")

  # TODO: IM-GAN

  def test_z(self, config, batch_z, dim):  # GAN
    # TODO:
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      return
    # load previous checkpoint
    if os.path.exists(self.checkpoint_path):
      self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for t in range(batch_z.shape[0]):
      model_z = batch_z[t:t+1]
      model_z = tf.convert_to_tensor(model_z)
      model_float = self.z2voxel(model_z)
      # img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
      # img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
      # img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
      # cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
      # cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
      # cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

      vertices, triangles = mcubes.marching_cubes(
          model_float, self.sampling_threshold)
      vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
      # vertices = self.optimize_mesh(vertices,model_z)
      write_ply(config.sample_dir+"/"+"out" +
                str(t)+".ply", vertices, triangles)

      print("[sample Z]")
