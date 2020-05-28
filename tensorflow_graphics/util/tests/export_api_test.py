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
"""Tests for export API functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import test_case


class ExportAPITest(test_case.TestCase):

  def test_get_functions_and_classes(self):
    """Tests that get_functions_and_classes does not raise an exception."""
    try:
      export_api.get_functions_and_classes()
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_get_modules(self):
    """Tests that get_modules does not raise an exception."""
    try:
      export_api.get_modules()
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))


if __name__ == "__main__":
  test_case.main()
