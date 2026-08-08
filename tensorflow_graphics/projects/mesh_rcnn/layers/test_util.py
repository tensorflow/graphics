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
"""Utility functions for testing of Mesh R-CNN layers."""



def calc_conv_out_spatial_shape(in_width,
                                in_height,
                                kernel_size=3,
                                stride=1,
                                padding=0):
  """Computes the output size of a Conv2D layer."""
  out_width = ((in_width - kernel_size + 2 * padding)/stride) + 1
  out_height = ((in_height - kernel_size + 2 * padding)/stride) + 1
  return int(out_width), int(out_height)


def calc_deconv_out_spatial_shape(in_width,
                                  in_height,
                                  kernel_size,
                                  stride,
                                  padding=0):
  """Computes the output size of a Conv2DTranspose layer."""
  out_width = stride * (in_width - 1) + kernel_size - 2 * padding
  out_height = stride * (in_height - 1) + kernel_size - 2 * padding
  return int(out_width), int(out_height)
