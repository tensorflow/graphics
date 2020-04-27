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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "modelnet40",
    srcs = ["__init__.py"],
    data = [":url_checksums"],
    deps = [
        # google internal package dependency 6
        # google internal package dependency 1,
        "//third_party/py/tensorflow_datasets",
    ],
)

filegroup(
    name = "url_checksums",
    srcs = glob(["checksums/*.txt"]),
)

filegroup(
    name = "fake_examples",
    srcs = glob([
        "fakes/modelnet40_ply_hdf5_2048/**",
    ]),
)

py_test(
    name = "modelnet40_test",
    srcs = ["modelnet40_test.py"],
    data = [":fake_examples"],
    python_version = "PY3",
    deps = [
        ":modelnet40",
        "//third_party/py/absl/logging",
    ],
)

py_binary(
    name = "modelnet40_makefakes",
    srcs = ["modelnet40_makefakes.py"],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":modelnet40",
        "//third_party/py/tensorflow_datasets",
    ],
)

py_binary(
    name = "modelnet40_run",
    srcs = ["modelnet40_run.py"],
    python_version = "PY3",
    deps = [
        ":modelnet40",
        "//third_party/py/absl:app",
        # google internal package dependency 4,
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//tensorflow_graphics/datasets",
    ],
)

py_binary(
    name = "modelnet40_checksums",
    srcs = ["modelnet40_checksums.py"],
    python_version = "PY3",
    deps = [
        ":modelnet40",
        "//third_party/py/absl:app",
        # google internal package dependency 4,
        "//tensorflow_graphics/datasets",
    ],
)
