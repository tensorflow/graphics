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
from PIL import Image
import numpy as np

from sklearn.utils import shuffle


class KittiDataset(object):
  r""" Kitti Instance dataset.

  Args:
      dataset_folder (str): path to the KITTI dataset
      img_size (int): size of the cropped images
      transform (list): list of transformations applied to the images
      return_idx (bool): wether to return index
  """

  def __init__(self, dataset_folder, batch_size=32, shuffle=False,
               random_state=None, img_size=224, transform=None,
               return_idx=False):
    self.dataset = []
    self.batch_size = batch_size
    self.shuffle = shuffle
    if random_state is None:
      random_state = np.random.RandomState(1234)
    self.random_state = random_state
    self._index = 0
    self.img_size = img_size
    self.img_path = os.path.join(dataset_folder, 'image_2')
    crop_path = os.path.join(dataset_folder, 'cropped_images')
    self.cropped_images = []
    for folder in os.listdir(crop_path):
      folder_path = os.path.join(crop_path, folder)
      for file_name in os.listdir(folder_path):
        current_file_path = os.path.join(folder_path, file_name)
        self.cropped_images.append(current_file_path)

    self.transform = transform
    self.return_idx = return_idx

    for idx in range(len(self.cropped_images)):
      cropped_img_r = tf.io.read_file(self.cropped_images[idx])
      cropped_img = tf.image.decode_image(
          cropped_img_r, channels=3, dtype=tf.float32)
      cropped_img = tf.image.resize(cropped_img, [224, 224])
      cropped_img /= 255.0

      idx = tf.convert_to_tesnor(idx)

      data = {
          'inputs': cropped_img,
          'idx': idx,
      }

      self.dataset.append(data)

    self._reset()

  def get_model_dict(self, idx):
    model_dict = {
        'model': str(idx),
        'category': 'kitti',
    }
    return model_dict

  # def get_model(self, idx):
  #     ''' Returns the model.

  #     Args:
  #         idx (int): ID of data point
  #     '''
  #     f_name = os.path.basename(self.cropped_images[idx])[:-4]
  #     return f_name

  def __len__(self):
    ''' Returns the length of the dataset.
    '''
    n = len(self.dataset)
    b = self.batch_size
    return n // b + bool(n % b)

  def __next__(self):
    ''' Returns an item of the dataset.
    Args:
        idx (int): ID of data point
    '''
    if self._index >= len(self.dataset):
      self._reset()
      raise StopIteration()

    indexes = self.dataset[self._index:
                           (self._index + self.batch_size)]

    self._index += self.batch_size
    return indexes

  def _reset(self):
    if self.shuffle:
      self.dataset = shuffle(self.dataset,
                             random_state=self.random_state)
    self._index = 0


class OnlineProductDataset(object):
  r""" Stanford Online Product Dataset.

  Args:
      dataset_folder (str): path to the dataset dataset
      img_size (int): size of the cropped images
      classes (list): list of classes
      max_number_imgs (int): maximum number of images
      return_idx (bool): wether to return index
      return_category (bool): wether to return category
  """

  def __init__(self, dataset_folder, batch_size=32, shuffle=False,
               random_state=None, img_size=224, classes=['chair'],
               max_number_imgs=1000, return_idx=False, return_category=False):

    self.dataset = []
    self.batch_size = batch_size
    self.shuffle = shuffle
    if random_state is None:
      random_state = np.random.RandomState(1234)
    self.random_state = random_state
    self._index = 0

    self.img_size = img_size
    self.dataset_folder = dataset_folder
    self.transform = lambda image: tf.image.resize(
        image, [img_size, img_size]) / 255.0
    self.class_id = {}
    self.metadata = []

    for i, cl in enumerate(classes):
      self.metadata.append({'name': cl})
      self.class_id[cl] = i
      cl_names = np.loadtxt(
          os.path.join(dataset_folder, cl+'_final.txt'), dtype=np.str)
      cl_names = cl_names[:max_number_imgs]
      att = np.vstack(
          (cl_names, np.full_like(cl_names, cl))).transpose(1, 0)
      if i > 0:
        self.file_names = np.vstack((self.file_names, att))
      else:
        self.file_names = att

    self.return_idx = return_idx
    self.return_category = return_category

    for idx in range(len(self.file_names)):
      f = os.path.join(
          self.dataset_folder,
          self.file_names[idx, 1]+'_final',
          self.file_names[idx, 0])

      img_in = Image.open(f)
      img = Image.new("RGB", img_in.size)
      img.paste(img_in)
      img = tf.keras.preprocessing.image.img_to_array(img)

      cl_id = tf.convert_to_tensor(
          self.class_id[self.file_names[idx, 1]])
      idx = tf.convert_to_tesnor(idx)

      if self.transform:
        img = self.transform(img)

      data = {
          'inputs': img,
      }

      if self.return_idx:
        data['idx'] = idx

      if self.return_category:
        data['category'] = cl_id

      self.dataset.append(data)

    self._reset()

  def get_model_dict(self, idx):
    category_id = self.class_id[self.file_names[idx, 1]]

    model_dict = {
        'model': str(idx),
        'category': category_id
    }
    return model_dict

  def get_model(self, idx):
    ''' Returns the model.

    Args:
        idx (int): ID of data point
    '''
    file_name = os.path.basename(self.file_names[idx, 0])[:-4]
    return file_name

  def __len__(self):
    ''' Returns the length of the dataset.
    '''
    n = len(self.dataset)
    b = self.batch_size
    return n // b + bool(n % b)

  def __next__(self):
    ''' Returns an item of the dataset.
    Args:
        idx (int): ID of data point
    '''
    if self._index >= len(self.dataset):
      self._reset()
      raise StopIteration()

    indexes = self.dataset[self._index:
                           (self._index + self.batch_size)]

    self._index += self.batch_size
    return indexes

  def _reset(self):
    if self.shuffle:
      self.dataset = shuffle(self.dataset,
                             random_state=self.random_state)
    self._index = 0


IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'
)


class ImageDataset(object):
  r""" Cars Dataset.

  Args:
      dataset_folder (str): path to the dataset dataset
      img_size (int): size of the cropped images
      transform (list): list of transformations applied to the data points
  """

  def __init__(self, dataset_folder, batch_size=32, shuffle=False,
               random_state=None, img_size=224, return_idx=False):
    """

    Arguments:
        dataset_folder (path): path to the KITTI dataset
        img_size (int): required size of the cropped images
        return_idx (bool): wether to return index
    """

    self.dataset = []
    self.batch_size = batch_size
    self.shuffle = shuffle
    if random_state is None:
      random_state = np.random.RandomState(1234)
    self.random_state = random_state
    self._index = 0

    self.img_size = img_size
    self.img_path = dataset_folder
    self.file_list = os.listdir(self.img_path)
    self.file_list = [
        f for f in self.file_list
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    ]
    self.len = len(self.file_list)
    self.transform = lambda image: tf.image.resize(
        image, [224, 224]) / 255.0

    self.return_idx = return_idx

    for idx in range(len(self.file_list)):
      f = os.path.join(self.img_path, self.file_list[idx])
      img_in = Image.open(f)
      img = Image.new("RGB", img_in.size)
      img.paste(img_in)
      img = tf.keras.preprocessing.image.img_to_array(img)

      if self.transform:
        img = self.transform(img)

      idx = tf.convert_to_tesnor(idx)

      data = {
          'inputs': img,
      }

      if self.return_idx:
        data['idx'] = idx

      self.dataset.append(data)

    self._reset()

  def get_model(self, idx):
    ''' Returns the model.

    Args:
        idx (int): ID of data point
    '''
    f_name = os.path.basename(self.file_list[idx])
    f_name = os.path.splitext(f_name)[0]
    return f_name

  def get_model_dict(self, idx):
    f_name = os.path.basename(self.file_list[idx])
    model_dict = {
        'model': f_name
    }
    return model_dict

  def __len__(self):
    ''' Returns the length of the dataset.
    '''
    n = len(self.dataset)
    b = self.batch_size
    return n // b + bool(n % b)

  def __next__(self):
    ''' Returns an item of the dataset.
    Args:
        idx (int): ID of data point
    '''
    if self._index >= len(self.dataset):
      self._reset()
      raise StopIteration()

    indexes = self.dataset[self._index:
                           (self._index + self.batch_size)]

    self._index += self.batch_size
    return indexes

  def _reset(self):
    if self.shuffle:
      self.dataset = shuffle(self.dataset,
                             random_state=self.random_state)
    self._index = 0
