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

import tensorflow as tf

import yaml
from im2mesh import data
from im2mesh import onet  # r2n2, psgn, pix2mesh, dmc
from im2mesh import preprocess

method_dict = {
    "onet": onet,
    # "r2n2": r2n2,
    # "psgn": psgn,
    # "pix2mesh": pix2mesh,
    # "dmc": dmc,
}
'''
TODO: torchvision transforms codes -> tensorflow
'''


# General config
def load_config(path, default_path=None):
  """ Loads config file.

  Args:
      path (str): path to config file
      default_path (bool): whether to use default path
  """
  # Load configuration from file itself
  with open(path, "r") as f:
    cfg_special = yaml.load(f)

  # Check if we should inherit from a config
  inherit_from = cfg_special.get("inherit_from")

  # If yes, load this config first as default
  # If no, use the default_path
  if inherit_from is not None:
    cfg = load_config(inherit_from, default_path)
  elif default_path is not None:
    with open(default_path, "r") as f:
      cfg = yaml.load(f)
  else:
    cfg = dict()

  # Include main configuration
  update_recursive(cfg, cfg_special)

  return cfg


def update_recursive(dict1, dict2):
  """ Update two config dictionaries recursively.

  Args:
      dict1 (dict): first dictionary to be updated
      dict2 (dict): second dictionary which entries should be used

  """
  for k, v in dict2.items():
    if k not in dict1:
      dict1[k] = dict()
    if isinstance(v, dict):
      update_recursive(dict1[k], v)
    else:
      dict1[k] = v


# Models
def get_model(cfg, dataset=None):
  """ Returns the model instance.

  Args:
      cfg (dict): config dictionary
      dataset (dataset): dataset
  """
  method = cfg["method"]
  model = method_dict[method].config.get_model(cfg, dataset=dataset)
  return model


# Trainer
def get_trainer(model, optimizer, cfg):
  """ Returns a trainer instance.

  Args:
      model (nn.Module): the model which is used
      optimizer (optimizer): pytorch optimizer
      cfg (dict): config dictionary
  """
  method = cfg["method"]
  trainer = method_dict[method].config.get_trainer(model, optimizer, cfg)
  return trainer


# Generator for final mesh extraction
def get_generator(model, cfg):
  """ Returns a generator instance.

  Args:
      model (nn.Module): the model which is used
      cfg (dict): config dictionary
  """
  method = cfg["method"]
  generator = method_dict[method].config.get_generator(model, cfg)
  return generator


# Datasets
def get_dataset(mode, cfg, batch_size, shuffle, repeat_count, epoch,
                return_idx=False, return_category=False):
  """ Returns the dataset.

  Args:
      model (nn.Module): the model which is used
      cfg (dict): config dictionary
      return_idx (bool): whether to include an ID field
  """
  method = cfg["method"]
  dataset_type = cfg["data"]["dataset"]
  dataset_folder = cfg["data"]["path"]
  categories = cfg["data"]["classes"]

  # Get split
  splits = {
      "train": cfg["data"]["train_split"],
      "val": cfg["data"]["val_split"],
      "test": cfg["data"]["test_split"],
  }

  split = splits[mode]

  # Create dataset
  if dataset_type == "Shapes3D":
    # Dataset fields
    # Method specific fields (usually correspond to output)
    fields = method_dict[method].config.get_data_fields(mode, cfg)
    # Input fields
    inputs_field = get_inputs_field(mode, cfg)
    if inputs_field is not None:
      fields["inputs"] = inputs_field

    if return_idx:
      fields["idx"] = data.IndexField()

    if return_category:
      fields["category"] = data.CategoryField()

    dataset = data.Shapes3dDataset(
        dataset_folder,
        fields,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        repeat_count=repeat_count,
        epoch=epoch,
        categories=categories,
    )
  elif dataset_type == "kitti":
    dataset = data.KittiDataset(dataset_folder,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                img_size=cfg["data"]["img_size"],
                                return_idx=return_idx)
  elif dataset_type == "online_products":
    dataset = data.OnlineProductDataset(
        dataset_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        img_size=cfg["data"]["img_size"],
        classes=cfg["data"]["classes"],
        max_number_imgs=cfg["generation"]["max_number_imgs"],
        return_idx=return_idx,
        return_category=return_category,
    )
  elif dataset_type == "images":
    dataset = data.ImageDataset(
        dataset_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        img_size=cfg["data"]["img_size"],
        return_idx=return_idx,
    )
  else:
    raise ValueError('Invalid dataset "%s"' % cfg["data"]["dataset"])

  return dataset


def get_inputs_field(mode, cfg):
  """ Returns the inputs fields.

  Args:
      mode (str): the mode which is used
      cfg (dict): config dictionary
  """
  input_type = cfg["data"]["input_type"]
  with_transforms = cfg["data"]["with_transforms"]

  if input_type is None:
    inputs_field = None
  elif input_type == "img":
    transform = None
    if mode == "train" and cfg["data"]["img_augment"]:
      # resize_op = transforms.RandomResizedCrop(cfg["data"]["img_size"],(0.75, 1.0), (1.0, 1.0))
      def preprocess(image):
        # image = tf.image.crop_and_resize(
        #     image, crop_size=cfg["data"]["img_size"])  # CHECK
        image = tf.image.resize(
            image, [cfg["data"]["img_size"], cfg["data"]["img_size"]])
        image /= 255.0
        return image

      transform = preprocess
    else:
      def preprocess(image):
        # image = image[tf.newaxis, ...]
        image = tf.image.resize(
            image, [cfg["data"]["img_size"], cfg["data"]["img_size"]])
        image /= 255.0
        return image

      transform = preprocess

    # transform = transforms.Compose([
    #     resize_op,
    #     transforms.ToTensor(),
    # ])

    with_camera = cfg["data"]["img_with_camera"]

    if mode == "train":
      random_view = True
    else:
      random_view = False

    inputs_field = data.ImagesField(
        cfg["data"]["img_folder"],
        transform,
        with_camera=with_camera,
        random_view=random_view,
    )
  elif input_type == "pointcloud":
    def preprocess(points):
      output = data.SubsamplePointcloud(
          cfg["data"]["pointcloud_n"])(points)
      output = data.PointcloudNoise(
          cfg["data"]["pointcloud_noise"])(output)
      return output

    transform = preprocess
    with_transforms = cfg["data"]["with_transforms"]
    inputs_field = data.PointCloudField(cfg["data"]["pointcloud_file"],
                                        transform,
                                        with_transforms=with_transforms)
  elif input_type == "voxels":
    inputs_field = data.VoxelsField(cfg["data"]["voxels_file"])
  elif input_type == "idx":
    inputs_field = data.IndexField()
  else:
    raise ValueError("Invalid input type (%s)" % input_type)
  return inputs_field


def get_preprocessor(cfg, dataset=None):
  """ Returns preprocessor instance.

  Args:
      cfg (dict): config dictionary
      dataset (dataset): dataset
  """
  p_type = cfg["preprocessor"]["type"]
  cfg_path = cfg["preprocessor"]["config"]
  model_file = cfg["preprocessor"]["model_file"]

  if p_type == "psgn":
    preprocessor = preprocess.PSGNPreprocessor(
        cfg_path=cfg_path,
        pointcloud_n=cfg["data"]["pointcloud_n"],
        dataset=dataset,
        model_file=model_file,
    )
  elif p_type is None:
    preprocessor = None
  else:
    raise ValueError("Invalid Preprocessor %s" % p_type)

  return preprocessor
