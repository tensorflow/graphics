# Copyright 2020 Google LLC
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
# Tensorflow-Graphics, a library for differentiable computer graphics and geometry.

load("//configlang/ncl/build_defs:ncl.bzl", "ncl_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_binary", "pytype_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

pytype_library(
    name = "tensorflow_graphics",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    srcs_version = "PY2AND3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow_graphics/datasets",
        "//third_party/py/tensorflow_graphics/geometry",
        "//third_party/py/tensorflow_graphics/image",
        "//third_party/py/tensorflow_graphics/math",
        "//third_party/py/tensorflow_graphics/nn",
        "//third_party/py/tensorflow_graphics/notebooks",
        "//third_party/py/tensorflow_graphics/rendering",
        "//third_party/py/tensorflow_graphics/util",
        "//third_party/py/tensorflow_graphics/util:doc",
        "//third_party/py/tensorflow_graphics/util:export_api",
    ],
)

pytype_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        "//pyglib:file_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow_docs/api_generator:generate_lib",
        "//third_party/py/tensorflow_graphics",
        "//third_party/py/tensorflow_graphics/rendering/opengl:gl_stub_libs",  # Prevent linking issues.
        "//third_party/py/tensorflow_graphics/util:doc",
    ],
)

ncl_test(
    name = "blueprint_test",
    srcs = ["tensorflow_graphics.blueprint"],
    data = [
        "//devtools/blueprint/ncl:blueprint_file_ncl",
        "//devtools/blueprint/ncl:sanitizer",
        "//releasetools/rapid/ncl:rapid_config_ncl",
    ],
)
