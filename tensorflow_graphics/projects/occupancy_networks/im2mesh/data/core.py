# copyright 2020 the tensorflow authors
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#    https://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
""" NO COMMENT NOW"""


import os
import logging
import random
import yaml

import tensorflow as tf

logger = logging.getLogger(__name__)


# Fields
class Field(object):
  ''' Data fields class.
  '''

  def load(self, data_path, idx, category):
    ''' Loads a data point.
    Args:
        data_path (str): path to data file
        idx (int): index of data point
        category (int): index of category
    '''
    raise NotImplementedError

  def check_complete(self, files):
    ''' Checks if set is complete.
    Args:
        files: files
    '''
    raise NotImplementedError


# Dataset and Dataloader
class Shapes3dDataset(object):
  ''' 3D Shapes dataset class.
  '''

  def __init__(self, dataset_folder, fields, split=None, batch_size=32,
               shuffle=False, repeat_count=1, epoch=10,
               categories=None, no_except=True, transform=None):
    ''' Initialization of the the 3D shape dataset.
    Args:
        dataset_folder (str): dataset folder
        fields (dict): dictionary of fields
        split (str): which split is used
        batch_size (int):
        shuffle (bool): shuffle or not
        repeat_count (int): epoch
        categories (list): list of categories to use
        no_except (bool): no exception
        transform (callable): transformation applied to data points
    '''
    # Attributes
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.dataset_folder = dataset_folder
    self.repeat_count = repeat_count
    self.fields = fields
    self.no_except = no_except
    self.transform = transform
    self.epoch = epoch

    # If categories is None, use all subfolders
    if categories is None:
      categories = os.listdir(dataset_folder)
      categories = [c for c in categories
                    if os.path.isdir(os.path.join(dataset_folder, c))]

    # Read metadata file
    metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

    if os.path.exists(metadata_file):
      with open(metadata_file, 'r') as f:
        self.metadata = yaml.load(f)
    else:
      self.metadata = {
          c: {'id': c, 'name': 'n/a'} for c in categories
      }

    # Set index
    for c_idx, c in enumerate(categories):
      self.metadata[c]['idx'] = c_idx

    # Get all models
    self.models = []
    for c_idx, c in enumerate(categories):
      subpath = os.path.join(dataset_folder, c)
      if not os.path.isdir(subpath):
        logger.warning('Category %s does not exist in dataset.', c)

      split_file = os.path.join(subpath, split + '.lst')
      with open(split_file, 'r') as f:
        models_c = f.read().split('\n')

      self.models += [
          {'category': c, 'model': m}
          for m in models_c
      ]
    # print(self.models)

    random.shuffle(self.models)

    print("ShapeNet3D dataset __init__ complete")

  def __len__(self):
    ''' Returns the length of the dataset.
    '''
    n = len(self.models)
    b = self.batch_size
    return n // b + bool(n % b)

  def generator(self):
    ''' Returns an item of the dataset.
    Args:
        idx (int): ID of data point
    '''

    for idx in range(len(self.models)):
      category = self.models[idx]['category']
      model = self.models[idx]['model']
      c_idx = self.metadata[category]['idx']

      model_path = os.path.join(self.dataset_folder, category, model)
      data = {}

      for field_name, field in self.fields.items():
        try:
          field_data = field.load(model_path, idx, c_idx)
        except Exception:
          if self.no_except:
            logger.warning(
                'Error occured when loading field %s of model %s',
                field_name, model
            )
            return None
          else:
            raise

        if isinstance(field_data, dict):
          for k, v in field_data.items():
            if k is None:
              data[field_name] = v
            else:
              data['%s.%s' % (field_name, k)] = v
        else:
          data[field_name] = field_data

      if self.transform is not None:
        data = self.transform(data)

      yield data

  def dataset_keys(self):
    category = self.models[0]['category']
    model = self.models[0]['model']
    c_idx = self.metadata[category]['idx']

    model_path = os.path.join(self.dataset_folder, category, model)
    data = {}

    for field_name, field in self.fields.items():
      try:
        field_data = field.load(model_path, 0, c_idx)
      except Exception:
        if self.no_except:
          logger.warning(
              'Error occured when loading field %s of model %s',
              field_name, model
          )
          return None
        else:
          raise

      if isinstance(field_data, dict):
        for k, v in field_data.items():
          if k is None:
            data[field_name] = v
          else:
            data['%s.%s' % (field_name, k)] = v
      else:
        data[field_name] = field_data

    return data.keys()

  def loader(self):
    loader = tf.data.Dataset.from_generator(
        self.generator,
        output_types={k: tf.float32 for k in self.dataset_keys()})
    loader.apply(tf.data.experimental.shuffle_and_repeat(
        self.batch_size * 3, self.epoch))
    loader = loader.batch(batch_size=self.batch_size)
    loader = loader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return loader

  def get_model_dict(self, idx):
    return self.models[idx]

  def test_model_complete(self, category, model):
    ''' Tests if model is complete.
    Args:
        model (str): modelname
    '''
    model_path = os.path.join(self.dataset_folder, category, model)
    files = os.listdir(model_path)
    for field_name, field in self.fields.items():
      if not field.check_complete(files):
        logger.warning('Field "%s" is incomplete: %s',
                       field_name, model_path)
        return False

    return True
