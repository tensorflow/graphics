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
"""Pointcloud Tracking."""
from os import path

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.nasa.lib import datasets
from tensorflow_graphics.projects.nasa.lib import model_utils
from tensorflow_graphics.projects.nasa.lib import models
from tensorflow_graphics.projects.nasa.lib import utils

tf.disable_eager_execution()

flags = tf.compat.v1.app.flags
logging = tf.logging
tf.logging.set_verbosity(tf.logging.INFO)

utils.define_flags()
flags.mark_flag_as_required("joint_data")
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.random.set_random_seed(20200823)
  np.random.seed(20200823)

  input_fn = datasets.get_dataset("test", FLAGS)
  batch = input_fn(None).make_one_shot_iterator().get_next()

  # Extracting a motion sequence in a data dict
  data = {}
  with tf.Session() as sess:
    while True:
      try:
        batch_val = sess.run(batch)
        key = batch_val["name"][0]
        data[key] = batch_val
        data[key]["vert"] = (
            data[key]["vert"] +
            np.random.normal(0, 5e-3, data[key]["vert"].shape))
      except tf.errors.OutOfRangeError:
        break
    sorted_keys = sorted(data.keys())

  # Parse relevant parameters for theta optimization
  trans_range = FLAGS.trans_range
  n_dims = FLAGS.n_dims
  n_translate = 3 if n_dims == 3 else 2
  n_rotate = 6 if n_dims == 3 else 2

  # Set up parameters and place holders.
  tf.reset_default_graph()
  accum_mat_holder = tf.placeholder(tf.float32,
                                    [FLAGS.n_parts, n_dims + 1, n_dims + 1])
  pt_holder = tf.placeholder(tf.float32, [1, 1, None, FLAGS.n_dims])
  weight_holder = tf.placeholder(tf.float32, [1, 1, None, FLAGS.n_parts])
  loss_holder = tf.placeholder(tf.float32, [])
  glue_loss_holder = tf.placeholder(tf.float32, [])
  iou_holder = tf.placeholder(tf.float32, [])
  id_transform = model_utils.get_identity_transform(n_translate, n_rotate,
                                                    FLAGS.n_parts)
  theta = tf.Variable(id_transform, trainable=True, name="pose_var")

  # Compute transformation matrix and joints according to theta
  temp_mat = model_utils.get_transform_matrix(theta, trans_range, n_translate,
                                              n_rotate, n_dims)
  if FLAGS.left_trans:
    trans_mat = tf.matmul(
        tf.reshape(accum_mat_holder,
                   [1, 1, FLAGS.n_parts, n_dims + 1, n_dims + 1]),
        tf.reshape(temp_mat, [1, 1, FLAGS.n_parts, n_dims + 1, n_dims + 1]))
  else:
    trans_mat = tf.matmul(
        tf.reshape(temp_mat, [1, 1, FLAGS.n_parts, n_dims + 1, n_dims + 1]),
        tf.reshape(accum_mat_holder,
                   [1, 1, FLAGS.n_parts, n_dims + 1, n_dims + 1]))
  r = trans_mat[..., :n_dims, :n_dims]
  t = trans_mat[..., :n_dims, -1:]
  r_t = tf.transpose(r, [0, 1, 2, 4, 3])
  t_0 = -tf.matmul(r_t, t)
  joint_trans = tf.concat(
      [tf.concat([r_t, t_0], axis=-1), trans_mat[..., -1:, :]], axis=-2)
  joint_trans = tf.reshape(joint_trans, [FLAGS.n_parts, n_dims + 1, n_dims + 1])
  inv_first_frame_trans = data[sorted_keys[0]]["transform"].reshape(
      [FLAGS.n_parts, n_dims + 1, n_dims + 1])
  joint_trans = tf.matmul(joint_trans, inv_first_frame_trans)
  first_frame_joint = data[sorted_keys[0]]["joint"].reshape(
      [FLAGS.n_parts, n_dims, 1])
  first_frame_joint = tf.concat(
      [first_frame_joint,
       tf.ones_like(first_frame_joint[..., :1, :])], axis=-2)
  joint = tf.matmul(joint_trans, first_frame_joint)[..., :-1, 0]

  if FLAGS.glue_w > 0.:
    with tf.io.gfile.GFile(FLAGS.joint_data, "rb") as cin:
      connectivity = np.load(cin)
    end_points = data[sorted_keys[0]]["joint"].reshape([FLAGS.n_parts, n_dims])
    first_frame_trans = data[sorted_keys[0]]["transform"].reshape(
        [FLAGS.n_parts, n_dims + 1, n_dims + 1])
    glue_loss = utils.compute_glue_loss(
        connectivity, end_points,
        tf.reshape(trans_mat, [FLAGS.n_parts, n_dims + 1, n_dims + 1]),
        first_frame_trans, joint, FLAGS)
  else:
    glue_loss = tf.constant(0, dtype=tf.float32)

  # Set up computation graph
  model_fn = models.get_model(FLAGS)
  batch_holder = {
      "transform": trans_mat,
      "joint": joint,
      "point": pt_holder,
      "weight": weight_holder,
  }
  if FLAGS.gradient_type == "vanilla":
    interface = utils.vanilla_theta_gradient(model_fn, batch_holder, FLAGS)
  elif FLAGS.gradient_type == "reparam":
    interface = utils.reparam_theta_gradient(model_fn, batch_holder, FLAGS)

  # Parse content of the interface
  latent_holder, latent, occ, rec_loss = interface
  if FLAGS.glue_w > 0:
    loss = rec_loss + glue_loss * FLAGS.glue_w
  else:
    loss = rec_loss
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(FLAGS.theta_lr)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(
        loss,
        var_list=[theta],
        global_step=global_step,
        name="optimize_theta",
    )
    reset_op = tf.variables_initializer(
        [theta, global_step] + optimizer.variables(), name="reset_button")

  tf.summary.scalar("Loss", loss_holder)
  tf.summary.scalar("IoU", iou_holder)
  tf.summary.scalar("Glue", glue_loss_holder)
  summary_op = tf.summary.merge_all()

  # Load checkpoint and run optimization
  assignment_map = {
      "shape/": "shape/",
  }
  tf.train.init_from_checkpoint(FLAGS.train_dir, assignment_map)
  init_op = tf.global_variables_initializer()
  with tf.summary.FileWriter(FLAGS.train_dir) as summary_writer:
    with tf.Session() as sess:
      sess.run(init_op)
      accum_mat = data[sorted_keys[0]]["transform"].reshape(
          [FLAGS.n_parts, n_dims + 1, n_dims + 1])
      accum_iou = 0.
      example_cnt = 0
      for frame_id, k in enumerate(sorted_keys):
        data_example = data[k]
        feed_dict = {
            pt_holder: data_example["vert"],
            weight_holder: data_example["weight"],
            accum_mat_holder: accum_mat,
        }
        loss_val, loss_glue_val = utils.optimize_theta(feed_dict, loss,
                                                       reset_op, train_op,
                                                       rec_loss, glue_loss,
                                                       sess, k, FLAGS)
        iou = utils.compute_iou(sess, feed_dict, latent_holder, pt_holder,
                                latent, occ[:, -1:], data_example["points"],
                                data_example["labels"], FLAGS)
        accum_iou += iou
        example_cnt += 1
        utils.save_mesh(
            sess,
            feed_dict,
            latent_holder,
            pt_holder,
            latent,
            occ,
            data_example,
            FLAGS,
            pth="tracked_{}".format(FLAGS.gradient_type))
        utils.save_pointcloud(
            data_example,
            FLAGS,
            pth="pointcloud_{}".format(FLAGS.gradient_type))

        summary = sess.run(summary_op, {
            loss_holder: loss_val,
            iou_holder: iou,
            glue_loss_holder: loss_glue_val
        })
        summary_writer.add_summary(summary, frame_id)
        summary_writer.flush()

        temp_mat_val = sess.run(temp_mat)
        if FLAGS.left_trans:
          accum_mat = np.matmul(
              accum_mat,
              temp_mat_val.reshape([FLAGS.n_parts, n_dims + 1, n_dims + 1]))
        else:
          accum_mat = np.matmul(
              temp_mat_val.reshape([FLAGS.n_parts, n_dims + 1, n_dims + 1]),
              accum_mat)

      with tf.io.gfile.GFile(
          path.join(FLAGS.train_dir, "tracked_{}".format(FLAGS.gradient_type),
                    "iou.txt"), "w") as iout:
        iout.write("{}\n".format(accum_iou / example_cnt))


if __name__ == "__main__":
  tf.app.run(main)
