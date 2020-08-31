import tensorflow as tf

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
        self.gf_dim*8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_2 = tf.keras.layers.Dense(
        self.gf_dim*8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_3 = tf.keras.layers.Dense(
        self.gf_dim*8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_4 = tf.keras.layers.Dense(
        self.gf_dim*4, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_5 = tf.keras.layers.Dense(
        self.gf_dim*2, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_6 = tf.keras.layers.Dense(
        self.gf_dim*1, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02), bias_initializer='zeros')
    self.linear_7 = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(
        1e-5, 0.02), bias_initializer='zeros')

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


class resnet_block(tf.keras.Model):
  def __init__(self, dim_in, dim_out):
    super(resnet_block, self).__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    if self.dim_in == self.dim_out:
      self.conv_1 = tf.keras.layers.Conv2D(
          self.dim_out, 3, strides=1, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
      self.bn_1 = tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05)
      self.conv_2 = tf.keras.layers.Conv2D(
          self.dim_out, 3, strides=1, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
      self.bn_2 = tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05)
    else:
      self.conv_1 = tf.keras.layers.Conv2D(
          self.dim_out, 3, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
      self.bn_1 = tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05)
      self.conv_2 = tf.keras.layers.Conv2D(
          self.dim_out, 3, strides=1, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
      self.bn_2 = tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05)
      self.conv_s = tf.keras.layers.Conv2D(
          self.dim_out, 1, strides=2, padding='valid', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
      self.bn_s = tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05)

  def __call__(self, input, training=False):
    if self.dim_in == self.dim_out:
      output = self.bn_1(self.conv_1(input), training=training)
      output = leaky_relu(output, 0.02)
      output = self.bn_2(self.conv_2(output), training=training)
      output = output + input
      output = leaky_relu(output, 0.02)
    else:
      output = self.bn_1(self.conv_1(input), training=training)
      output = leaky_relu(output, 0.02)
      output = self.bn_2(self.conv_2(output), training=training)
      input_ = self.bn_s(self.conv_s(input), training=training)
      output = output + input_
      output = leaky_relu(output, 0.02)
    return output


class img_encoder(tf.keras.Model):
  def __init__(self, img_ef_dim, z_dim):
    super(img_encoder, self).__init__()
    self.img_ef_dim = img_ef_dim
    self.z_dim = z_dim
    self.conv_0 = tf.keras.layers.Conv2D(
        self.img_ef_dim, 7, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.bn_0 = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
    self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
    self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim * 2)
    self.res_4 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 2)
    self.res_5 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 4)
    self.res_6 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 4)
    self.res_7 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 8)
    self.res_8 = resnet_block(self.img_ef_dim * 8, self.img_ef_dim * 8)

    self.conv_9 = tf.keras.layers.Conv2D(
        self.img_ef_dim * 8, 4, strides=2, padding='same', use_bias=False, kernel_initializer=tf.initializers.GlorotUniform())
    self.bn_9 = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    self.conv_10 = tf.keras.layers.Conv2D(
        self.z_dim, 4, strides=1, padding='valid', use_bias=True, kernel_initializer=tf.initializers.GlorotUniform())

  def __call__(self, view, training=False):

    layer_0 = self.bn_0(self.conv_0(1-view), training=training)
    layer_0 = leaky_relu(layer_0, 0.02)

    layer_1 = self.res_1(layer_0, training=training)
    layer_2 = self.res_2(layer_1, training=training)

    layer_3 = self.res_3(layer_2, training=training)
    layer_4 = self.res_4(layer_3, training=training)

    layer_5 = self.res_5(layer_4, training=training)
    layer_6 = self.res_6(layer_5, training=training)

    layer_7 = self.res_7(layer_6, training=training)
    layer_8 = self.res_8(layer_7, training=training)

    layer_9 = self.bn_9(self.conv_9(layer_8), training=training)
    layer_9 = leaky_relu(layer_9, 0.02)

    layer_10 = self.conv_10(layer_9)
    layer_10 = tf.reshape(layer_10, [-1, self.z_dim])
    layer_10 = tf.keras.activations.sigmoid(layer_10)

    return layer_10


class im_network(tf.keras.Model):
  def __init__(self, img_ef_dim, gf_dim, z_dim, point_dim):
    super(im_network, self).__init__()
    self.img_ef_dim = img_ef_dim
    self.gf_dim = gf_dim
    self.z_dim = z_dim
    self.point_dim = point_dim
    self.img_encoder = img_encoder(self.img_ef_dim, self.z_dim)
    self.generator = generator(
        self.z_dim, self.point_dim, self.gf_dim)
    self.generator.trainable = False

  def __call__(self, inputs, z_vector, point_coord, training=False):
    if training:
      z_vector = self.img_encoder(inputs, training=training)
      net_out = None
    else:
      if inputs is not None:
        z_vector = self.img_encoder(inputs, training=training)
      if z_vector is not None and point_coord is not None:
        net_out = self.generator(
            point_coord, z_vector, training=training)
      else:
        net_out = None

    return z_vector, net_out


class IM_SVR(object):
  def __init__(self, config):

    self.input_size = 64  # input voxel grid size

    self.img_ef_dim = 64
    self.gf_dim = 128
    self.z_dim = 256
    self.point_dim = 3

    # actual batch size
    self.shape_batch_size = 64

    self.view_size = 137
    self.crop_size = 128
    self.view_num = 24
    self.crop_edge = self.view_size - self.crop_size
    self.test_idx = 23

    self.dataset_name = config.dataset
    self.dataset_load = self.dataset_name + '_train'
    if not config.train:
      self.dataset_load = self.dataset_name + '_test'
    self.checkpoint_dir = config.checkpoint_dir
    self.data_dir = config.data_dir

    data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
    if os.path.exists(data_hdf5_name):
      data_dict = h5py.File(data_hdf5_name, 'r')
      offset_x = int(self.crop_edge/2)
      offset_y = int(self.crop_edge/2)
      # reshape to NCHW
      self.data_pixels = np.reshape(data_dict['pixels'][:, :, offset_y:offset_y+self.crop_size,
                                                        offset_x:offset_x+self.crop_size], [-1, self.view_num, 1, self.crop_size, self.crop_size])
    else:
      print("error: cannot load "+data_hdf5_name)
      exit(0)
    if config.train:
      dataz_hdf5_name = self.checkpoint_dir+'/' + \
          self.modelAE_dir+'/'+self.dataset_name+'_train_z.hdf5'
      if os.path.exists(dataz_hdf5_name):
        print("data_hdf5_name:{}".format(dataz_hdf5_name))
        dataz_dict = h5py.File(dataz_hdf5_name, 'r')
        self.data_zs = dataz_dict['zs'][:]
      else:
        print("error: cannot load "+dataz_hdf5_name)
        exit(0)
      if len(self.data_zs) != len(self.data_pixels):
        print("error: len(self.data_zs) != len(self.data_pixels)")
        print(len(self.data_zs), len(self.data_pixels))
        exit(0)

    # build model
    self.im_network = im_network(
        self.img_ef_dim, self.gf_dim, self.z_dim, self.point_dim)
    # print params
    # for param_tensor in self.im_network.state_dict():
    #	print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
    # TODO:
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=0.999)
    self.ckpt = tf.train.Checkpoint(
        model=self.im_network, optimizer=self.optimizer)

    # TODO:
    self.max_to_keep = 10
    self.checkpoint_path = os.path.join(
        self.checkpoint_dir, self.model_dir)
    self.checkpoint_name = 'IM_SVR.model'
    self.checkpoint_manager_list = [None] * self.max_to_keep
    self.checkpoint_manager_pointer = 0
    self.checkpoint_AE_path = os.path.join(
        self.checkpoint_dir, self.modelAE_dir)
    self.checkpoint_AE_name = 'IM_AE.model'
    # loss

    def network_loss(pred_z, gt_z):
      return tf.reduce_mean(tf.math.pow(pred_z-gt_z, 2))

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
    return "{}_svr_{}".format(
        self.dataset_name, self.crop_size)

  @property
  def modelAE_dir(self):
    return "{}_ae_{}".format(
        self.dataset_name, self.input_size)

  def train(self, config):
    # load AE weights
    # checkpoint_txt = os.path.join(self.checkpoint_AE_path, "checkpoint")
    if os.path.exists(self.checkpoint_AE_path):
      # print("###############################")
      # print(tf.train.latest_checkpoint(self.checkpoint_AE_path))
      # print("###############################")
      # self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_AE_path))
      self.im_network.generator.load_weights(
          '/home/mil/kasuga/IM-NET_tf/checkpoint/all_vox256_img_ae_64/IM_AE.model_generator64-400.ckpt')
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      exit(-1)

    shape_num = len(self.data_pixels)
    batch_index_list = np.arange(shape_num)

    print("\n\n----------net summary----------")
    print("training samples   ", shape_num)
    print("-------------------------------\n\n")

    start_time = time.time()
    assert config.epoch == 0 or config.iteration == 0
    training_epoch = config.epoch + int(config.iteration/shape_num)
    batch_num = int(shape_num/self.shape_batch_size)

    # self.im_network.train()
    for epoch in tqdm(range(0, training_epoch)):
      np.random.shuffle(batch_index_list)
      avg_loss = 0
      avg_num = 0
      for idx in range(batch_num):
        dxb = batch_index_list[idx *
                               self.shape_batch_size:(idx+1)*self.shape_batch_size]

        which_view = np.random.randint(self.view_num)
        batch_view = self.data_pixels[dxb, which_view].astype(
            np.float32)/255.0
        batch_zs = self.data_zs[dxb]

        batch_view = batch_view.transpose(0, 2, 3, 1)

        batch_view = tf.convert_to_tensor(batch_view)
        batch_zs = tf.convert_to_tensor(batch_zs)

        # TODO:
        with tf.GradientTape() as tape:
          z_vector, _ = self.im_network(
              batch_view, None, None, training=True)
          err = self.loss(z_vector, batch_zs)
        grad_im_network = tape.gradient(
            err, self.im_network.img_encoder.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grad_im_network, self.im_network.img_encoder.trainable_weights))

        avg_loss += err
        avg_num += 1
      print("Epoch: [%2d/%2d] time: %4.4f, loss: %.8f" % (epoch,
                                                          training_epoch, time.time() - start_time, avg_loss/avg_num))
      if epoch % 10 == 9:
        self.test_1(config, "train_"+str(epoch))
      if epoch % 100 == 99:
        if not os.path.exists(self.checkpoint_path):
          os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(
            self.checkpoint_path, self.checkpoint_name+"-"+str(epoch)+".ckpt")
        self.checkpoint_manager_pointer = (
            self.checkpoint_manager_pointer+1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
          if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
            os.remove(
                self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        # TODO:
        # torch.save(self.im_network.state_dict(), save_dir)
        self.ckpt.save(save_dir)

        save_dir_imgencoder = os.path.join(
            self.checkpoint_path, self.checkpoint_name+"_imgencoder"+"-"+str(epoch)+".ckpt")
        self.im_network.img_encoder.save_weights(save_dir_imgencoder)
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
    save_dir = os.path.join(
        self.checkpoint_path, self.checkpoint_name+"-"+str(training_epoch)+".ckpt")
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
    save_dir_imgencoder = os.path.join(
        self.checkpoint_path, self.checkpoint_name+"_imgencoder"+"-"+str(epoch)+".ckpt")
    self.im_network.img_encoder.save_weights(save_dir_imgencoder)

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
    # self.im_network.eval()
    t = np.random.randint(len(self.data_pixels))
    model_float = np.zeros(
        [self.frame_grid_size+2, self.frame_grid_size+2, self.frame_grid_size+2], np.float32)
    batch_view = self.data_pixels[t:t+1,
                                  self.test_idx].astype(np.float32)/255.0
    batch_view = batch_view.transpose(0, 2, 3, 1)
    batch_view = tf.convert_to_tensor(batch_view)
    z_vector, _ = self.im_network(
        batch_view, None, None, training=False)
    for i in range(multiplier):
      for j in range(multiplier):
        for k in range(multiplier):
          minib = i*multiplier2+j*multiplier+k
          point_coord = self.coords[minib:minib+1]
          _, net_out = self.im_network(
              None, z_vector, point_coord, training=False)
          # net_out = torch.clamp(net_out, min=0, max=1)
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

    # self.im_network.eval()
    for t in range(config.start, min(len(self.data_pixels), config.end)):
      batch_view_ = self.data_pixels[t:t+1,
                                     self.test_idx].astype(np.float32)/255.0
      batch_view = batch_view.transpose(0, 2, 3, 1)
      batch_view = tf.convert_to_tensor(batch_view_)
      model_z, _ = self.im_network(
          batch_view, None, None, training=False)
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
      self.ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path))
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      return

    # self.im_network.eval()
    for t in range(config.start, min(len(self.data_pixels), config.end)):
      batch_view_ = self.data_pixels[t:t+1,
                                     self.test_idx].astype(np.float32)/255.0
      batch_view = batch_view.transpose(0, 2, 3, 1)
      batch_view = tf.convert_to_tensor(batch_view_)
      model_z, _ = self.im_network(
          batch_view, None, None, training=False)
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
