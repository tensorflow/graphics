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
workspace(name = "tensorflow_graphics")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.0/bazel-skylib-1.0.0.tar.gz",
    sha256 = "e72747100a8b6002992cc0bf678f6279e71a3fd4a88cab3371ace6c73432be30",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "9d359cc1b508082d8ba309ba085da6ecec85e7a4d5bd08f8db9666ee39a85529",
    strip_prefix = "rules_closure-0.9.0",
    url = "https://github.com/bazelbuild/rules_closure/archive/0.9.0.zip",
)

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
)

