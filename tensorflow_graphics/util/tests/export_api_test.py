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

import sys
import types

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import test_case


def sample_public_function():
  """Sample public function for export_api testing."""


class _SamplePrivateClass:
  """Sample private class for export_api testing."""


class ExportAPITest(test_case.TestCase):

  def test_get_functions_and_classes(self):
    """Tests that get_functions_and_classes returns public symbols."""
    exported = export_api.get_functions_and_classes()
    self.assertIn("ExportAPITest", exported)
    self.assertIn("sample_public_function", exported)
    self.assertNotIn("_SamplePrivateClass", exported)

  def test_get_modules(self):
    """Tests that get_modules returns public direct submodules."""
    self.assertEqual(export_api.get_modules(), [])

    pkg_name = "_tfg_export_api_test_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__dict__["export_api"] = export_api
    pkg.__dict__["child"] = types.ModuleType(pkg_name + ".child")
    pkg.__dict__["_private"] = types.ModuleType(pkg_name + "._private")
    pkg.__dict__["other"] = types.ModuleType("unrelated_pkg.other")

    def _call():
      return export_api.get_modules()

    sys.modules[pkg_name] = pkg
    try:
      call_in_pkg = types.FunctionType(_call.__code__, pkg.__dict__)
      self.assertEqual(call_in_pkg(), ["child"])
    finally:
      sys.modules.pop(pkg_name, None)


if __name__ == "__main__":
  test_case.main()
