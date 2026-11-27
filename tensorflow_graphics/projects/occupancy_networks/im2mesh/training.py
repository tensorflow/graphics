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

# from im2mesh import icp
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class BaseTrainer(object):
  ''' Base trainer class.
  '''

  def evaluate(self, val_loader):
    ''' Performs an evaluation.
    Args:
        val_loader (dataloader): pytorch dataloader
    '''
    eval_list = defaultdict(list)

    for data in tqdm(val_loader):
      eval_step_dict = self.eval_step(data)

      for k, v in eval_step_dict.items():
        eval_list[k].append(v)

    eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
    return eval_dict

  def train_step(self, *args, **kwargs):
    ''' Performs a training step.
    '''
    raise NotImplementedError

  def eval_step(self, *args, **kwargs):
    ''' Performs an evaluation step.
    '''
    raise NotImplementedError

  def visualize(self, *args, **kwargs):
    ''' Performs  visualization.
    '''
    raise NotImplementedError
