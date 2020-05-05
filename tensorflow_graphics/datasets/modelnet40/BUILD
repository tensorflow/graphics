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
# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "modelnet40",
    srcs = ["__init__.py"],
    data = [":url_checksums"],
    deps = [
        # google internal package dependency 11
        # google internal package dependency 6
        # google internal package dependency 1,
        # google internal package dependency 5,
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
    size = "small",
    srcs = ["modelnet40_test.py"],
    data = [":fake_examples"],
    deps = [
        ":modelnet40",
        # google internal package dependency 5,
        # google internal package dependency 12
    ],
)

py_binary(
    name = "modelnet40_makefakes",
    srcs = ["modelnet40_makefakes.py"],
    python_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":modelnet40",
        # google internal package dependency 5,
    ],
)

py_binary(
    name = "modelnet40_run",
    srcs = ["modelnet40_run.py"],
    python_version = "PY3",
    deps = [
        ":modelnet40",
        # google internal package dependency 13,
        # google internal package dependency 4,
        # google internal package dependency 5,
    ],
)

# TODO(atagliasacchi): test when non-headless enviroment available
# py_binary(
#     name = "modelnet40_show",
#     srcs = ["modelnet40_show.py"],
#     python_version = "PY3",
#     deps = [
#         ":modelnet40",
#         requirement("absl-py"),
#         requirement("matplotlib"),
#         # google internal package dependency 5,
#     ],
# )

py_binary(
    name = "modelnet40_checksums",
    srcs = ["modelnet40_checksums.py"],
    python_version = "PY3",
    deps = [
        ":modelnet40",
        # google internal package dependency 13,
        # google internal package dependency 4,
        # google internal package dependency 5,
    ],
)
