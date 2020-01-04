from .device import *
from .camera_type import *
from .camera import *
from .shape import *
from .texture import *
from .material import *
from .area_light import *
from .object import *
from .envmap import *
from .scene import *
from .image import *
from .load_obj import load_obj
from .load_mitsuba import load_mitsuba
from .transform import *
from .utils import *
from .channels import *
from .sampler_type import *
from .render_utils import *
from .render_tensorflow import *
from tensorflow_graphics.util import export_api as _export_api

try:
    import redner
except ImportError:
    print("Warning: redner is not installed when you import pyredner_tensorflow."
          " Please install redner, by following instructions at https://github.com/BachiLi/redner/wiki"
          " or by using pip install redner-gpu or pip install redner.")
from packaging import version
import tensorflow as tf
import re

if version.parse(re.sub('[^0-9.]', '', tf.__version__)) < version.parse('2.0.0'):
    raise ImportError("Detected tensorflow version {},"
                      "redner only supports TensorFlow version >= 2.0.0".format(tf.__version__))
import os.path

if tf.__cxx11_abi_flag__ == 0:
    __data_ptr_module = tf.load_op_library(
        os.path.join(os.path.dirname(redner.__file__), 'libredner_tf_data_ptr_no_cxx11_abi.so'))
else:
    assert (tf.__cxx11_abi_flag__ == 1)
    __data_ptr_module = tf.load_op_library(
        os.path.join(os.path.dirname(redner.__file__), 'libredner_tf_data_ptr_cxx11_abi.so'))


def data_ptr(tensor):
    addr_as_uint64 = __data_ptr_module.data_ptr(tensor)
    return int(addr_as_uint64)


# API contains submodules of tensorflow_graphics.models.
__all__ = _export_api.get_modules()
