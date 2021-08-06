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
"""Texture module."""

# pylint: disable=g-import-not-at-top
from tensorflow_graphics.util.doc import _import_tfg_docs
if _import_tfg_docs():
  from tensorflow_graphics.rendering.texture import texture_map
  from tensorflow_graphics.rendering.texture import mipmap
  from tensorflow_graphics.util import export_api as _export_api

  # API contains submodules of tensorflow_graphics.rendering.
  __all__ = _export_api.get_modules()
# pylint: enable=g-import-not-at-top
