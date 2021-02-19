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
"""Preprocess."""
# python3
import inspect

from google3.third_party.tensorflow_models.object_detection.core import preprocessor


def preprocess(tensor_dict,
               preprocess_options,
               func_arg_map=None,
               preprocess_vars_cache=None):
  """Preprocess sample before the batch is created.

  Inspired by:
  google3.third_party.tensorflow_models.object_detection.core.preprocessor
  preprocessor

  Args:
    tensor_dict: dictionary that contains images, boxes, and can contain other
      things as well.
    preprocess_options: It is a list of tuples, where each tuple contains a
      function and a dictionary that contains arguments and their values.
    func_arg_map: mapping from preprocessing functions to arguments that they
      expect to receive and return. For each key (which is the function), there
      should be a list with the keys in the tensor_dict. The order of the list
      should be the same as the order in the function arguments. Its values can
      also be None if the input argument is not used.
    preprocess_vars_cache: PreprocessorCache object that records previously
      performed augmentations. Updated in-place. If this function is called
      multiple times with the same non-null cache, it will perform
      deterministically.

  Returns:
    tensor_dict: which contains the preprocessed images, bounding boxes, etc.

  Raises:
    ValueError: (a) If the functions passed to Preprocess
                    are not in func_arg_map.
                (b) If the arguments that a function needs
                    do not exist in tensor_dict.

  The output of the function should return the tensors that will be assigned to
  tensor_dict keys using the same order mapping as the input in func_arg_map.
  Example to resize image:
    preprocess_options = [(preprocessor.resize_image, {'new_height': 600,
    'new_width': 1024})]
    func_arg_map = {preprocessor.resize_to_range:
    (fields.InputDataFields.image)}
  """
  if func_arg_map is None:
    func_arg_map = preprocessor.get_default_func_arg_map()

  # Preprocess inputs based on preprocess_options
  for option in preprocess_options:
    func, params = option
    if func not in func_arg_map:
      raise ValueError('The function %s does not exist in func_arg_map' %
                       (func.__name__))
    arg_names = func_arg_map[func]
    if isinstance(arg_names[0], (list, tuple)):
      arg_names_input, arg_names_output = arg_names
    else:
      arg_names_input, arg_names_output = arg_names, arg_names
    arg_names_input = [a if a in tensor_dict else None for a in arg_names_input]
    # for a in arg_names_input:
    #   if a is not None and a not in tensor_dict:
    #     raise ValueError('The function %s requires argument %s' %
    #                      (func.__name__, a))

    def get_arg(key):
      return tensor_dict[key] if key is not None else None

    args = [get_arg(a) for a in arg_names_input]
    if preprocess_vars_cache is not None:
      arg_spec = inspect.getfullargspec(func)
      if 'preprocess_vars_cache' in arg_spec.args:
        params['preprocess_vars_cache'] = preprocess_vars_cache

    results = func(*args, **params)
    if not isinstance(results, (list, tuple)):
      results = (results,)
    # Removes None args since the return values will not contain those.
    # arg_names = [
    #     arg_name for arg_name in arg_names_output if arg_name is not None
    # ]
    for res, arg_name in zip(results, arg_names_output):
      if res is not None:
        tensor_dict[arg_name] = res

  return tensor_dict
