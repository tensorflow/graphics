#Copyright 2019 Google LLC
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
"""Script to generate external api_docs for tf-graphics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
import tensorflow_graphics as tfg

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/graphics_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics",
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "graphics/api_docs/python",
                    "Path prefix in the _toc.yaml")


def main(_):
  doc_generator = generate_lib.DocGenerator(
      root_title="Tensorflow Graphics",
      py_modules=[("tfg", tfg)],
      base_dir=os.path.dirname(tfg.__file__),
      search_hints=FLAGS.search_hints,
      code_url_prefix=FLAGS.code_url_prefix,
      site_path=FLAGS.site_path)

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
