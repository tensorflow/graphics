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
"""Transforms."""
import functools

from cvx2 import latest as cv2
import numpy as np
import tensorflow as tf
from tensorflow_graphics.projects.points_to_3Dobjects.utils import image as image_utils

from google3.third_party.tensorflow_models.object_detection.core import preprocessor

tf_data_augmentation = preprocessor

LIGHTING_EIGVAL = (0.2141788, 0.01817699, 0.00341571)
LIGHTING_EIGVEC = ((-0.58752847, -0.69563484, 0.4134035),
                   (-0.5832747, 0.00994535, -0.8122141),
                   (-0.560893, 0.7183267, 0.41158938))


def bgr_to_rgb(image):
  return image[..., ::-1]


def rgb_to_bgr(image):
  return image[..., ::-1]


def brightness(image, variance):
  alpha = 1 + tf.random.uniform(
      [1], dtype=tf.float32, minval=-variance, maxval=variance)[0]
  return image * alpha


def contrast(image, image_grayscale_mean, variance):
  alpha = 1 + tf.random.uniform(
      [1], dtype=tf.float32, minval=-variance, maxval=variance)[0]
  return image * alpha + image_grayscale_mean * (1 - alpha)


def saturation(image, image_grayscale, variance):
  alpha = 1 + tf.random.uniform(
      [1], dtype=tf.float32, minval=-variance, maxval=variance)[0]
  return image * alpha + image_grayscale * (1 - alpha)


def lighting(image,
             alpha_std=0.1,
             eigval=LIGHTING_EIGVAL,
             eigvec=LIGHTING_EIGVEC):
  alpha = tf.random.normal([3], stddev=alpha_std, dtype=tf.float32)
  return image + tf.tensordot(
      tf.constant(eigvec), tf.constant(eigval) * alpha, axes=((1,), (0,)))


def color_augmentations(image, variance=0.4):
  """Color augmentations."""
  if variance:
    print(variance)
  image_grayscale = tf.image.rgb_to_grayscale(bgr_to_rgb(image))
  image_grayscale_mean = tf.math.reduce_mean(
      image_grayscale, axis=[-3, -2, -1], keepdims=True)
  brightness_fn = functools.partial(brightness, variance=variance)
  contrast_fn = functools.partial(
      contrast, image_grayscale_mean=image_grayscale_mean, variance=variance)
  saturation_fn = functools.partial(
      saturation, image_grayscale=image_grayscale, variance=variance)
  function_order = tf.random.shuffle([0, 1, 2])
  ii = tf.constant(0)

  def _apply_fn(image, ii):
    tmp_ii = function_order[ii]
    image = tf.switch_case(
        tmp_ii, {
            0: lambda: brightness_fn(image),
            1: lambda: contrast_fn(image),
            2: lambda: saturation_fn(image)
        })
    ii = ii + 1
    return image, ii

  (image, _) = tf.while_loop(lambda image, ii: tf.less(ii, 3),
                             _apply_fn(image, ii),
                             [image, ii])

  image = lighting(image)
  return image


def subtract_mean_and_normalize(image, means, std, random=False):
  if len(means) != len(std):
    raise ValueError('len(means) and len(std) must match')
  image = image / 255
  if random:
    image = color_augmentations(image)
  image = (image - tf.constant(means)) / tf.constant(std)
  return image


def _get_image_border(border, size):
  i = tf.constant(1)
  cond = lambda i: tf.math.less_equal(size - border // i, border // i)
  body = lambda i: tf.multiply(i, 2)
  r = tf.while_loop(cond, body, [i])
  return border // r[0]


def compute_image_size_affine_transform(original_image_size,
                                        input_image_size,
                                        padding_keep_size=127,
                                        random=False,
                                        random_side_scale_range=None):
  """Computer affine transform."""
  if input_image_size is None:
    input_h = tf.bitwise.bitwise_or(original_image_size[-2],
                                    padding_keep_size) + 1
    input_w = tf.bitwise.bitwise_or(original_image_size[-1],
                                    padding_keep_size) + 1
    input_size = tf.cast(tf.stack([input_w, input_h]), tf.float32)
    side_size = tf.cast(tf.stack([input_w, input_h]), tf.float32)
    center = tf.cast(
        tf.stack([original_image_size[-1] // 2, original_image_size[-2] // 2]),
        tf.float32)
  else:
    input_size = tf.cast(tf.stack(input_image_size), tf.float32)
    max_side = tf.reduce_max(original_image_size[-2:])
    side_size = tf.cast(tf.stack([max_side, max_side]), tf.float32)
    image_shape = tf.cast(original_image_size, tf.float32)
    center = tf.stack([image_shape[-1] / 2., image_shape[-2] / 2.])

  if random:
    assert random_side_scale_range is not None, (
        'Random random_side_scale_range has to be provided when computing '
        'random affine transformation!')
    scales = tf.range(*random_side_scale_range)
    scale_ii = tf.random.categorical(
        tf.ones_like(scales)[None, ...], 1, dtype=tf.int32)[0, 0]
    side_size = side_size * scales[scale_ii]
    h_border = _get_image_border(128, original_image_size[-2])
    w_border = _get_image_border(128, original_image_size[-1])
    center_x = tf.random.uniform([1],
                                 dtype=tf.int32,
                                 minval=w_border,
                                 maxval=(original_image_size[-1] - w_border))[0]
    center_y = tf.random.uniform([1],
                                 dtype=tf.int32,
                                 minval=h_border,
                                 maxval=(original_image_size[-2] - h_border))[0]
    center = tf.cast(tf.stack([center_x, center_y]), tf.float32)

  return center, side_size, input_size


def affine_transform(image,
                     original_image_size,
                     bounding_boxes,
                     instance_masks,
                     image_size,
                     padding_keep_size=127,
                     transform_gt_annotations=False,
                     random=False,
                     random_side_scale_range=None,
                     random_flip_probability=False):
  """Affine transform."""
  # bounding_boxes: normalized coordinates with (ymin, xmin, ymax, xmax)
  center, side_size, input_size = compute_image_size_affine_transform(
      tf.shape(image)[:-1], image_size, padding_keep_size, random,
      random_side_scale_range)
  flipped = False
  if random:

    def _flip(flipped, image, center):
      flipped = tf.math.logical_not(flipped)
      image = image[..., ::-1, :]
      center = tf.tensor_scatter_nd_update(center, tf.constant(
          [[0]]), [tf.cast(tf.shape(image)[-2], center.dtype) - center[0]])
      return flipped, image, center

    def _no_flip(flipped, image, center):
      return flipped, image, center

    flipped, image, center = tf.cond(
        tf.random.uniform([1], dtype=tf.float32)[0] < random_flip_probability,
        lambda: _flip(flipped, image, center),
        lambda: _no_flip(flipped, image, center))

    if instance_masks is not None:
      def _flip_mask(mask):
        return mask[..., ::-1]

      def _no_flip_mask(mask):
        return mask

      instance_masks = tf.cond(
          flipped,
          lambda: _flip_mask(instance_masks),
          lambda: _no_flip_mask(instance_masks))

  # affine_transform_image_np(image, center, side_size, input_size)
  input_image_size_static = image.shape
  [
      image,
  ] = tf.py_function(affine_transform_image_np,
                     [image, center, side_size, input_size], [tf.float32])
  if len(input_image_size_static) == 4:
    image.set_shape([image.shape[0], None, None, image.shape[-1]])
  else:
    image.set_shape([None, None, image.shape[-1]])

  if transform_gt_annotations:
    bounding_boxes_shape = bounding_boxes.shape

    [
        bounding_boxes,
    ] = tf.py_function(_affine_transform_points_np, [
        bounding_boxes, original_image_size, center, side_size, input_size,
        flipped
    ], [tf.float32])
    bounding_boxes.set_shape(bounding_boxes_shape)

    if instance_masks is not None:
      instance_masks_size_static = instance_masks.shape
      [
          instance_masks,
      ] = tf.py_function(affine_transform_image_np, [
          instance_masks[..., None], center, side_size, input_size,
          cv2.INTER_NEAREST
      ], [tf.float32])
      if len(instance_masks_size_static) == 4:
        instance_masks.set_shape([instance_masks.shape[0], None, None, None, 1])
      else:
        instance_masks.set_shape([None, None, None, 1])
      instance_masks = instance_masks[..., 0]

    original_image_size = tf.cast(input_size, original_image_size.dtype)

  return image, original_image_size, bounding_boxes, instance_masks


def affine_transform_image_np(image,
                              center,
                              side_size,
                              input_size,
                              interpolation_mode=cv2.INTER_LINEAR):
  """Affine transform numpy."""
  input_w, input_h = input_size.numpy()[0], input_size.numpy()[1]
  trans_input = image_utils.get_affine_transform(center.numpy(),
                                                 side_size.numpy(), 0,
                                                 [input_w, input_h])
  image_np = image.numpy()
  if image_np.ndim >= 4:
    image_np_shape = image_np.shape
    image_np = np.reshape(image_np, (-1, *image_np_shape[-3:]))
    input_images = []
    for ii in range(image_np.shape[0]):
      warped_input_image = cv2.warpAffine(
          image_np[ii, ...],
          trans_input, (input_w, input_h),
          flags=interpolation_mode)
      if warped_input_image.ndim != image_np[ii, ...].ndim:
        warped_input_image = warped_input_image[..., None]
      input_images.append(warped_input_image)

    input_image = np.stack(input_images, axis=0)
    input_image = np.reshape(input_image,
                             (*image_np_shape[:-3], *input_image.shape[-3:]))
  else:
    input_image = cv2.warpAffine(
        image_np,
        trans_input, (input_w, input_h),
        flags=interpolation_mode)
  return input_image.astype(np.float32)


def _affine_transform_points_np(bounding_boxes, original_image_size, center,
                                side_size, output_size, flip):
  """Affine transform points."""
  bounding_boxes_np = bounding_boxes.numpy()
  bounding_boxes_shape = bounding_boxes_np.shape
  original_image_size_np = original_image_size.numpy()

  if len(bounding_boxes_shape) == 3:
    bounding_boxes_np = np.reshape(bounding_boxes_np, [-1, 4])

  h, w = original_image_size_np[:2]

  if flip:
    bounding_boxes_np[..., [1, 3]] = 1.0 - bounding_boxes_np[..., [3, 1]]

  bounding_boxes_np = bounding_boxes_np * [h, w, h, w]

  box_min = np.stack([bounding_boxes_np[:, 1], bounding_boxes_np[:, 0]], axis=1)
  box_max = np.stack([bounding_boxes_np[:, 3], bounding_boxes_np[:, 2]], axis=1)
  box_min = image_utils.transform_points(box_min, center.numpy(),
                                         side_size.numpy(), output_size.numpy(),
                                         False)
  box_max = image_utils.transform_points(box_max, center.numpy(),
                                         side_size.numpy(), output_size.numpy(),
                                         False)
  box_min = np.stack([box_min[:, 1], box_min[:, 0]], axis=1)
  box_max = np.stack([box_max[:, 1], box_max[:, 0]], axis=1)

  bounding_boxes_np = np.concatenate([box_min, box_max], axis=1)

  h, w = output_size.numpy()[:2]
  bounding_boxes_np = np.clip(bounding_boxes_np / [h, w, h, w], 0.0, 1.0)

  if len(bounding_boxes_shape) == 3:
    bounding_boxes_np = np.reshape(bounding_boxes_np, bounding_boxes_shape)

  return bounding_boxes_np.astype(np.float32)


def transform_predictions(points, center, scale, output_size):
  # transform_points_np(points, center, scale, output_size)
  [
      points,
  ] = tf.py_function(transform_points_np,
                     [points, center, scale, output_size, True], [tf.float32])
  return points


def transform_points_np(points, center, scale, output_size, inverse):
  new_points = image_utils.transform_points(points.numpy(), center.numpy(),
                                            scale.numpy(), output_size.numpy(),
                                            inverse)
  return new_points
