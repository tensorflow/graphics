# Copyright 2020 The TensorFlow Authors
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
""" NO COMMENT NOW"""


import os
import tensorflow as tf
import tensorflow_probability as tfp

from im2mesh.encoder import encoder_dict
from im2mesh.onet import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, dataset=None):
  ''' Return the Occupancy Network model.
  Args:
      cfg (dict): imported yaml config
      dataset (dataset): dataset
  '''
  decoder = cfg['model']['decoder']
  encoder = cfg['model']['encoder']
  encoder_latent = cfg['model']['encoder_latent']
  z_dim = cfg['model']['z_dim']
  c_dim = cfg['model']['c_dim']
  decoder_kwargs = cfg['model']['decoder_kwargs']
  encoder_kwargs = cfg['model']['encoder_kwargs']
  encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

  decoder = models.decoder_dict[decoder](
      z_dim=z_dim, c_dim=c_dim,
      **decoder_kwargs
  )

  if z_dim != 0:
    encoder_latent = models.encoder_latent_dict[encoder_latent](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **encoder_latent_kwargs
    )
  else:
    encoder_latent = None

  if encoder == 'idx':
    encoder = tf.keras.layers.Embedding(len(dataset), c_dim)
  elif encoder is not None:
    encoder = encoder_dict[encoder](
        c_dim=c_dim,
        **encoder_kwargs
    )
  else:
    encoder = None

  p0_z = get_prior_z(cfg)
  model = models.OccupancyNetwork(
      decoder, encoder, encoder_latent, p0_z
  )

  return model


def get_trainer(model, optimizer, cfg):
  ''' Returns the trainer object.
  Args:
      model (tf.keras.Model): the Occupancy Network model
      optimizer (optimizer): tf.keras.optimizers object
      cfg (dict): imported yaml config
  '''
  threshold = cfg['test']['threshold']
  out_dir = cfg['training']['out_dir']
  vis_dir = os.path.join(out_dir, 'vis')
  input_type = cfg['data']['input_type']

  trainer = training.Trainer(
      model, optimizer,
      input_type=input_type,
      vis_dir=vis_dir, threshold=threshold,
      eval_sample=cfg['training']['eval_sample'],
  )

  return trainer


def get_generator(model, cfg):
  ''' Returns the generator object.
  Args:
      model (tf.keras.Model): Occupancy Network model
      cfg (dict): imported yaml config
  '''
  preprocessor = config.get_preprocessor(cfg)

  generator = generation.Generator3D(
      model,
      threshold=cfg['test']['threshold'],
      resolution0=cfg['generation']['resolution_0'],
      upsampling_steps=cfg['generation']['upsampling_steps'],
      sample=cfg['generation']['use_sampling'],
      refinement_step=cfg['generation']['refinement_step'],
      simplify_nfaces=cfg['generation']['simplify_nfaces'],
      preprocessor=preprocessor,
  )
  return generator


def get_prior_z(cfg):
  ''' Returns prior distribution for latent code z.
  Args:
      cfg (dict): imported yaml config
  '''
  z_dim = cfg['model']['z_dim']
  p0_z = tfp.distributions.Normal(
      tf.zeros(z_dim), tf.ones(z_dim))

  return p0_z


def get_data_fields(mode, cfg):
  ''' Returns the data fields.
  Args:
      mode (str): the mode which is used
      cfg (dict): imported yaml config
  '''
  points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
  with_transforms = cfg['model']['use_camera']

  fields = {}
  fields['points'] = data.PointsField(
      cfg['data']['points_file'], points_transform,
      with_transforms=with_transforms,
      unpackbits=cfg['data']['points_unpackbits'],
  )

  if mode in ('val', 'test'):
    points_iou_file = cfg['data']['points_iou_file']
    voxels_file = cfg['data']['voxels_file']
    if points_iou_file is not None:
      fields['points_iou'] = data.PointsField(
          points_iou_file,
          with_transforms=with_transforms,
          unpackbits=cfg['data']['points_unpackbits'],
      )
    if voxels_file is not None:
      fields['voxels'] = data.VoxelsField(voxels_file)

  return fields
