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
"""Global flags to be used by various modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS
TFG_ADD_ASSERTS_TO_GRAPH = 'tfg_add_asserts_to_graph'

flags.DEFINE_boolean(
    TFG_ADD_ASSERTS_TO_GRAPH, False,
    'If True, calling tensorflow_graphics functions may add assert '
    'nodes to the graph where necessary.', short_name='tfg_debug')

# The util functions or classes are not exported.
__all__ = []
