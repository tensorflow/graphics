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

from tqdm import trange
from im2mesh.common import compute_iou, make_3d_grid
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
  """ Trainer object for the Occupancy Network.

  Args:
      model (nn.Module): Occupancy Network model
      optimizer (optimizer): pytorch optimizer object
      input_type (str): input type
      vis_dir (str): visualization directory
      threshold (float): threshold value
      eval_sample (bool): whether to evaluate samples

  """

  def __init__(
      self,
      model,
      optimizer,
      input_type="img",
      vis_dir=None,
      threshold=0.5,
      eval_sample=False,
  ):
    self.model = model
    self.optimizer = optimizer
    self.input_type = input_type
    self.vis_dir = vis_dir
    self.threshold = threshold
    self.eval_sample = eval_sample

    if vis_dir is not None and not os.path.exists(vis_dir):
      os.makedirs(vis_dir)
    # print("self.model.trainable_weight:{}".format(
    #     self.model.trainable_weights))

  def train_step(self, data):
    """ Performs a training step.

    Args:
        data (dict): data dictionary
    """
    with tf.GradientTape() as tape:
      loss = self.compute_loss(data, training=True)

    grads = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(
        zip(grads, self.model.trainable_weights))

    return loss

  def eval_step(self, data):
    """ Performs an evaluation step.

    Args:
        data (dict): data dictionary
    """
    threshold = self.threshold
    eval_dict = {}

    # Compute elbo
    points = data.get("points")
    occ = data.get("points.occ")

    inputs = data.get("inputs", tf.zeros([points.shape[0], 0]))
    voxels_occ = data.get("voxels")

    points_iou = data.get("points_iou")
    occ_iou = data.get("points_iou.occ")

    kwargs = {}

    elbo, rec_error, kl = self.model.compute_elbo(
        points, occ, inputs, training=False, **kwargs)

    eval_dict["loss"] = -float(tf.reduce_mean(elbo))
    eval_dict["rec_error"] = float(tf.reduce_mean(rec_error))
    eval_dict["kl"] = float(tf.reduce_mean(kl))

    # Compute iou
    batch_size = points.shape[0]

    p_out = self.model(points_iou, inputs,
                       sample=self.eval_sample, training=False, **kwargs)

    occ_iou_np = (occ_iou >= 0.5).numpy()
    occ_iou_hat_np = (p_out.probs_parameter() >= threshold).numpy()
    iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
    eval_dict["iou"] = float(iou)

    # Estimate voxel iou
    if voxels_occ is not None:
      points_voxels = make_3d_grid(
          (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3
      )
      points_voxels = tf.broadcast_to(
          points_voxels, [batch_size, *points_voxels.shape]
      )

      p_out = self.model(points_voxels, inputs,
                         sample=self.eval_sample, training=False, **kwargs)

      voxels_occ_np = (voxels_occ >= 0.5).numpy()
      occ_hat_np = (p_out.probs_parameter() >= threshold).numpy()
      iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

      eval_dict["iou_voxels"] = float(iou_voxels)

    return eval_dict

  def visualize(self, data):
    """ Performs a visualization step for the data.

    Args:
        data (dict): data dictionary
    """

    batch_size = data["points"].shape[0]
    inputs = data.get("inputs", tf.zeros([batch_size, 0]))

    shape = (32, 32, 32)
    p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape)  # CHECK
    p = tf.broadcast_to(p, [batch_size, *p.shape])

    kwargs = {}
    p_r = self.model(p, inputs, sample=self.eval_sample,
                     training=False, **kwargs)

    occ_hat = tf.reshape(p_r.probs_parameter(), [batch_size, *shape])
    voxels_out = (occ_hat >= self.threshold).numpy()

    for i in trange(batch_size):
      input_img_path = os.path.join(self.vis_dir, "%03d_in.png" % i)
      vis.visualize_data(inputs[i], self.input_type, input_img_path)
      vis.visualize_voxels(
          voxels_out[i], os.path.join(self.vis_dir, "%03d.png" % i)
      )

  def compute_loss(self, data, training=False):
    """ Computes the loss.

    Args:
        data (dict): data dictionary
    """
    p = data.get("points")
    occ = data.get("points.occ")
    inputs = data.get("inputs", tf.zeros([p.shape[0], 0]))

    kwargs = {}
    c = self.model.encode_inputs(inputs, training=training)
    q_z = self.model.infer_z(p, occ, c, training=training, **kwargs)
    # z = q_z.rsample()
    # reparameterize
    # mean = q_z.mean()
    mean = q_z.mean()
    logvar = tf.math.log(q_z.variance())
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * 0.5) + mean

    # KL-divergence
    kl = tf.reduce_sum(
        tfp.distributions.kl_divergence(q_z, self.model.p0_z), axis=-1
    )
    loss = tf.reduce_mean(kl)

    # General points
    logits = self.model.decode(p, z, c, training=training, **kwargs).logits
    # loss_i = F.binary_cross_entropy_with_logits(logits,
    #                                             occ,
    #                                             reduction="none")

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(occ, logits)
    loss = loss + tf.reduce_mean(tf.reduce_sum(loss_i, axis=-1))

    return loss
