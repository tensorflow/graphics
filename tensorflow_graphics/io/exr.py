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
"""Utility functions for reading and writing EXR image files as numpy arrays."""

# pylint: disable=c-extension-no-member

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Imath
import numpy as np
import OpenEXR


_np_to_exr = {
    np.float16: Imath.PixelType.HALF,
    np.float32: Imath.PixelType.FLOAT,
    np.uint32: Imath.PixelType.UINT,
}
_exr_to_np = dict(zip(_np_to_exr.values(), _np_to_exr.keys()))


def channels_to_ndarray(exr, channel_names):
  """Copies channels from an OpenEXR.InputFile into a numpy array.

  If the EXR image is of size (width, height), the result will be a numpy array
  of shape (height, width, len(channel_names)), where the last dimension holds
  the channels in the order they were specified in channel_names. The requested
  channels must all have the same datatype.

  Args:
    exr: An OpenEXR.InputFile that is already open.
    channel_names: A list of strings naming the channels to read.

  Returns:
    A numpy ndarray.

  Raises:
    ValueError: If the channels have different datatypes.
    RuntimeError: If a channel has an unknown type.
  """
  channels_header = exr.header()['channels']
  window = exr.header()['dataWindow']
  width = window.max.x - window.min.x + 1
  height = window.max.y - window.min.y + 1

  def read_channel(channel):
    """Reads a single channel from the EXR."""
    channel_type = channels_header[channel].type
    try:
      numpy_type = _exr_to_np[channel_type.v]
    except KeyError:
      raise RuntimeError('Unknown EXR channel type: %s' % str(channel_type))
    flat_buffer = np.frombuffer(exr.channel(channel), numpy_type)
    return np.reshape(flat_buffer, [height, width])

  channels = [read_channel(c) for c in channel_names]
  if any([channels[0].dtype != c.dtype for c in channels[1:]]):
    raise ValueError('Channels have mixed datatypes: %s' %
                     ', '.join([str(c.dtype) for c in channels]))
  # Stack the arrays so that the channels dimension is the last (fastest
  # changing) dimension.
  return np.stack(channels, axis=-1)


def read_exr(filename, channel_names=None):
  """Opens an EXR file and copies the requested channels into an ndarray.

  The Python OpenEXR wrapper uses a dictionary for the channel header, so the
  ordering of the channels in the underlying file is lost. If channel_names is
  not passed, this function orders the output channels with any present RGBA
  channels first, followed by the remaining channels in alphabetical order.
  By convention, RGBA channels are named 'R', 'G', 'B', 'A', so this function
  looks for those strings.

  Args:
    filename: The name of the EXR file.
    channel_names: A list of strings naming the channels to read. If None, all
      channels will be read.

  Returns:
    A numpy array containing the image data, and a list of the corresponding
      channel names.
  """
  exr = OpenEXR.InputFile(filename)
  if channel_names is None:
    remaining_channel_names = list(exr.header()['channels'].keys())
    conventional_rgba_names = ['R', 'G', 'B', 'A']
    present_rgba_names = []
    # Pulls out any present RGBA names in RGBA order.
    for name in conventional_rgba_names:
      if name in remaining_channel_names:
        present_rgba_names.append(name)
        remaining_channel_names.remove(name)
    channel_names = present_rgba_names + sorted(remaining_channel_names)

  return channels_to_ndarray(exr, channel_names), channel_names


def write_exr(filename, values, channel_names):
  """Writes the values in a multi-channel ndarray into an EXR file.

  Args:
    filename: The filename of the output file
    values: A numpy ndarray with shape [height, width, channels]
    channel_names: A list of strings with length = channels

  Raises:
    TypeError: If the numpy array has an unsupported type.
    ValueError: If the length of the array and the length of the channel names
      list do not match.
  """
  if values.shape[-1] != len(channel_names):
    raise ValueError(
        'Number of channels in values does not match channel names (%d, %d)' %
        (values.shape[-1], len(channel_names)))
  header = OpenEXR.Header(values.shape[1], values.shape[0])
  try:
    exr_channel_type = Imath.PixelType(_np_to_exr[values.dtype.type])
  except KeyError:
    raise TypeError('Unsupported numpy type: %s' % str(values.dtype))
  header['channels'] = {
      n: Imath.Channel(exr_channel_type) for n in channel_names
  }
  channel_data = [values[..., i] for i in range(values.shape[-1])]
  exr = OpenEXR.OutputFile(filename, header)
  exr.writePixels(
      dict((n, d.tobytes()) for n, d in zip(channel_names, channel_data)))
  exr.close()
