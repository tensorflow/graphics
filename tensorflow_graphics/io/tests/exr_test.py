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
"""Tests for exr.py."""

# pylint: disable=c-extension-no-member

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tempfile
from absl.testing import parameterized
import Imath
import numpy as np
import OpenEXR

from tensorflow_graphics.io import exr
from tensorflow_graphics.util import test_case


def _WriteMixedDatatypesExr(filename):
  header = OpenEXR.Header(64, 64)
  header['channels'] = {
      'uint': Imath.Channel(Imath.PixelType(Imath.PixelType.UINT)),
      'half': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
  }
  channel_data = {
      'uint': np.zeros([64, 64], dtype=np.uint32).tobytes(),
      'half': np.zeros([64, 64], dtype=np.float16).tobytes()
  }
  output = OpenEXR.OutputFile(filename, header)
  output.writePixels(channel_data)
  output.close()


def _MakeTestImage(num_channels, datatype):
  if num_channels == 1:
    test_channel_names = ['L']
  else:
    test_channel_names = ['R', 'G', 'B', 'A', 'P.x', 'P.y', 'P.z']
  image = np.zeros([64, 64, num_channels], dtype=datatype)
  for i in range(num_channels):
    # Add a block of color to each channel positioned by the channel index.
    image[i:i + 10, i * 2:i * 2 + 10, i] = i / float(num_channels)
  return image, test_channel_names[:num_channels]


class ExrTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('l', 1, np.float32, False), ('rgb', 3, np.float32, False),
      ('rgba', 4, np.float32, False), ('rgbaxyz', 7, np.float32, False),
      ('float16', 4, np.float16, False), ('unit32', 4, np.uint32, False),
      ('pass_expected_channels', 7, np.float32, True))
  def test_write_read_roundtrip(self, num_channels, datatype, pass_channels):
    expected_image, expected_channels = _MakeTestImage(num_channels, datatype)
    with tempfile.NamedTemporaryFile() as temp:
      exr.write_exr(temp.name, expected_image, expected_channels)
      image, channels = exr.read_exr(
          temp.name, expected_channels if pass_channels else None)

    self.assertEqual(expected_channels, channels)
    self.assertEqual(image.tolist(), expected_image.tolist())

  def test_reading_mixed_datatypes_fails(self):
    with tempfile.NamedTemporaryFile() as temp:
      _WriteMixedDatatypesExr(temp.name)
      with self.assertRaisesRegexp(ValueError, 'Channels have mixed datatypes'):
        _, _ = exr.read_exr(temp.name)

  def test_writing_with_array_channel_name_mismatch_fails(self):
    array_three_channels = np.zeros([64, 64, 3], dtype=np.float32)
    names_two_channels = ['A', 'B']
    with tempfile.NamedTemporaryFile() as temp:
      with self.assertRaisesRegexp(
          ValueError,
          'Number of channels in values does not match channel names'):
        exr.write_exr(temp.name, array_three_channels, names_two_channels)

  def test_writing_unsupported_numpy_type_fails(self):
    uint8_array = np.zeros([64, 64, 3], dtype=np.uint8)
    names = ['R', 'G', 'B']
    with tempfile.NamedTemporaryFile() as temp:
      with self.assertRaisesRegexp(TypeError, 'Unsupported numpy type'):
        exr.write_exr(temp.name, uint8_array, names)

  def test_reading_unknown_exr_type_fails(self):
    image, channels = _MakeTestImage(3, np.float16)
    with tempfile.NamedTemporaryFile() as temp:
      exr.write_exr(temp.name, image, channels)
      exr_file = OpenEXR.InputFile(temp.name)
      # Deliberately break the R channel header info. A mock InputFile is
      # required to override the header() method.
      header_dict = exr_file.header()
      header_dict['channels']['R'].type.v = -1  # Any bad value will do.
      make_mock_exr = collections.namedtuple('MockExr', ['header', 'channel'])
      mock_broken_exr = make_mock_exr(lambda: header_dict, exr_file.channel)
      with self.assertRaisesRegexp(RuntimeError, 'Unknown EXR channel type'):
        _ = exr.channels_to_ndarray(mock_broken_exr, ['R', 'G', 'B'])


if __name__ == '__main__':
  test_case.main()
