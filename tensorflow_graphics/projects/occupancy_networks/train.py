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
import datetime
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO

matplotlib.use('Agg')

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
  model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
  model_selection_sign = -1
else:
  raise ValueError('model_selection_mode must be '
                   'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

# Dataset
# specify path

train_dataset = config.get_dataset(
    'train', cfg, batch_size=batch_size,
    shuffle=True, repeat_count=1, epoch=100)
val_dataset = config.get_dataset(
    'val', cfg, batch_size=10, shuffle=False, repeat_count=1, epoch=1)
vis_dataset = config.get_dataset(
    'val', cfg, batch_size=12, shuffle=False, repeat_count=1, epoch=1)

train_loader = train_dataset.loader()
val_loader = val_dataset.loader()
vis_loader = vis_dataset.loader()

data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-08)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

checkpoint_io = CheckpointIO(model, optimizer, model_selection_sign, out_dir)

try:
  checkpoint_io.load('model')
except FileExistsError:
  print("start from scratch")

epoch_it = checkpoint_io.ckpt.epoch_it
it = checkpoint_io.ckpt.it
metric_val_best = checkpoint_io.ckpt.metric_val_best

trainer = config.get_trainer(model, optimizer, cfg)

# Hack because of previous bug in code
if metric_val_best == np.inf or metric_val_best == -np.inf:
  metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f' %
      (model_selection_metric, metric_val_best))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# log
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

while True:
  epoch_it.assign_add(1)

  for batch in train_loader:
    it.assign_add(1)
    loss = trainer.train_step(batch)

    # Print output
    if print_every > 0 and (it % print_every) == 0:
      print('[Epoch %02d] it=%03d, loss=%.4f' % (epoch_it, it, loss))
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=it)

    # Visualize output
    if visualize_every > 0 and (it % visualize_every) == 0:
      print('Visualizing')
      trainer.visualize(data_vis)

    # Save checkpoint
    if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
      print('Saving checkpoint')
      checkpoint_io.save('model/model.ckpt', epoch_it=epoch_it, it=it,
                         loss_val_best=metric_val_best)
    # Backup if necessary
    if (backup_every > 0 and (it % backup_every) == 0):
      print('Backup checkpoint')
      checkpoint_io.save('backup/model_%d.ckpt' % it, epoch_it=epoch_it, it=it,
                         loss_val_best=metric_val_best)
    # Run validation
    if validate_every > 0 and (it % validate_every) == 0:
      print("evaluate")
      eval_dict = trainer.evaluate(val_loader)
      print("eval_dict")
      metric_val = eval_dict[model_selection_metric]
      print('validation metric (%s): %.4f'
            % (model_selection_metric, metric_val))
      for k, v in eval_dict.items():
        with test_summary_writer.as_default():
          tf.summary.scalar('val/%s' % k, v, step=it)

      print("metric_val_best:{}".format(metric_val_best))
      if model_selection_sign * (metric_val - metric_val_best) > 0:
        metric_val_best.assign(metric_val)
        print('New best model (loss %.4f)' % metric_val_best)
        checkpoint_io.save('model_best/model_best.ckpt', epoch_it=epoch_it, it=it,
                           loss_val_best=metric_val_best)

    # Exit if necessary
    if exit_after > 0 and (time.time() - t0) >= exit_after:
      print('Time limit reached. Exiting.')
      checkpoint_io.save('model/model.ckpt', epoch_it=epoch_it, it=it,
                         loss_val_best=metric_val_best)
      exit(3)
