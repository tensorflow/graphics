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
"""API export functions used to create the automated documentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect


def get_functions_and_classes():
  """Extracts a list of public functions and classes for the API generation.

  Returns:
    A list of function and class names.
  """
  caller = inspect.stack()[1]
  module = inspect.getmodule(caller[0])
  return [
      obj_name for obj_name, obj in inspect.getmembers(module)
      if inspect.isfunction(obj) or
      inspect.isclass(obj) and not obj_name.startswith("_")
  ]


def get_modules():
  """Extracts a list of public modules for the API generation.

  Returns:
    A list of module names.
  """
  caller = inspect.stack()[1]
  module = inspect.getmodule(caller[0])
  return [
      obj_name for obj_name, obj in inspect.getmembers(module)
      if inspect.ismodule(obj) and obj.__name__.rsplit(".", 1)[0] ==
      module.__name__ and not obj_name.startswith("_")
  ]


# The util functions or classes are not exported.
__all__ = []
