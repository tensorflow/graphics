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
# See the License for the specific
""" use build_grid_ds op in python """

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

build_grid_ds_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_build_grid_ds_ops.so'))
build_grid_ds = build_grid_ds_ops.build_grid_ds
