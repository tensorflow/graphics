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

import argparse
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')

# Get configuration and basic arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

# Shorthands
out_dir = cfg['training']['out_dir']
out_file = os.path.join(out_dir, 'eval_full.pkl')
out_file_class = os.path.join(out_dir, 'eval.csv')

# Dataset
dataset = config.get_dataset(
    'test', cfg, batch_size=1, shuffle=False, repeat_count=1, epoch=1)
# Loader
dataloader = dataset.loader()

model = config.get_model(cfg, dataset=dataset)
dummy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-08)

checkpoint_io = CheckpointIO(model, dummy_optimizer, checkpoint_dir=out_dir)

try:
  checkpoint_io.load(cfg['test']['model_file'])
except FileExistsError:
  print('Model file does not exist. Exiting.')
  exit()

# Trainer
trainer = config.get_trainer(model, None, cfg)

eval_dicts = []
print('Evaluating networks...')

# Handle each dataset separately
for it, data in enumerate(tqdm(dataloader)):
  if data is None:
    print('Invalid data.')
    continue
  # Get index etc.
  # idx = data['idx'].item()
  idx = it

  try:
    model_dict = dataset.get_model_dict(idx)
  except AttributeError:
    model_dict = {'model': str(idx), 'category': 'n/a'}

  modelname = model_dict['model']
  category_id = model_dict['category']

  try:
    category_name = dataset.metadata[category_id].get('name', 'n/a')
  except AttributeError:
    category_name = 'n/a'

  eval_dict = {
      'idx': idx,
      'class id': category_id,
      'class name': category_name,
      'modelname': modelname,
  }
  eval_dicts.append(eval_dict)
  eval_data = trainer.eval_step(data)
  eval_dict.update(eval_data)


# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
