# TFG-PointCloud Custom Ops

This repository is based on the [TensorFlow-custom-op](https://github.com/tensorflow/custom-op) repository,
for a detailed guide about how to add an op we refer to this template.

## Setting up the environment
The c++-toolchains are dependent on the latest tensorflow-custom-op docker container, to set it up run
```bash
  docker pull tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16
  sudo docker run --gpus all --privileged -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:2.2.0-custom-op-gpu-ubuntu16
  root@docker: ./configure.sh
```

and answer the first question with yes `y`,  and sepcifiy the TensorFlow version.

## Building the PIP-Package

To compile the pip package run
``` bash
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
```

The package `.whl` is located in `artifacts/`, by default it should be a python3 package.

To install the package via pip run
```bash
  pip3 install artifacts/*.whl
```

To test out the package run
```bash
  cd ..
  python3 -c 'import tfg_custom_ops'
```
## Additional Information

You may use this software under the
[Apache 2.0 License](https://github.com/schellmi42/tensorflow_graphics_point_clouds/blob/master/LICENSE).

<!-- This is a guide for users who want to write custom c++ op for TensorFlow and distribute the op as a pip package. This repository serves as both a working example of the op building and packaging process, as well as a template/starting point for writing your own ops. The way this repository is set up allow you to build your custom ops from TensorFlow's pip package instead of building TensorFlow from scratch. This guarantee that the shared library you build will be binary compatible with TensorFlow's pip packages.

This guide currently supports Ubuntu and Windows custom ops, and it includes examples for both cpu and gpu ops.

Starting from Aug 1, 2019, nightly previews `tf-nightly` and `tf-nightly-gpu`, as well as
official releases `tensorflow` and `tensorflow-gpu` past version 1.14.0 are now built with a
different environment (Ubuntu 16.04 compared to Ubuntu 14.04, for example) as part of our effort to make TensorFlow's pip pacakges
manylinux2010 compatible. To help you building custom ops on linux, here we provide our toolchain in the format of a combination of a Docker image and bazel configurations.  Please check the table below for the Docker image name needed to build your custom ops.

|          |          CPU custom op          |          GPU custom op         |
|----------|:-------------------------------:|:------------------------------:|
| TF nightly  |    nightly-custom-op-ubuntu16   | nightly-custom-op-gpu-ubuntu16 |
| TF >= 2.1   |   2.1.0-custom-op-ubuntu16  |    2.1.0-custom-op-gpu-ubuntu16    |
| TF 1.5, 2.0 | custom-op-ubuntu16-cuda10.0 |       custom-op-gpu-ubuntu16       |
| TF <= 1.4   |        custom-op-ubuntu14       |     custom-op-gpu-ubuntu14     |


Note: all above Docker images have prefix `tensorflow/tensorflow:`

The bazel configurations are included as part of this repository.

## Build Example zero_out Op (CPU only)
If you want to try out the process of building a pip package for custom op, you can use the source code from this repository following the instructions below.

### For Windows Users
You can skip this section if you are not building on Windows. If you are building custom ops for Windows platform, you will need similar setup as building TensorFlow from source mentioned [here](https://www.tensorflow.org/install/source_windows). Additionally, you can skip all the Docker steps from the instructions below. Otherwise, the bazel commands to build and test custom ops stay the same.

### Setup Docker Container
You are going to build the op inside a Docker container. Pull the provided Docker image from TensorFlow's Docker hub and start a container.

Use the following command if the TensorFlow pip package you are building
against is not yet manylinux2010 compatible:
```bash
  docker pull tensorflow/tensorflow:custom-op-ubuntu14
  docker run -it tensorflow/tensorflow:custom-op-ubuntu14 /bin/bash
```
And the following instead if it is manylinux2010 compatible:

```bash
  docker pull tensorflow/tensorflow:custom-op-ubuntu16
  docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

Inside the Docker container, clone this repository. The code in this repository came from the [Adding an op](https://www.tensorflow.org/extend/adding_an_op) guide.
```bash
git clone https://github.com/tensorflow/custom-op.git
cd custom-op
```

### Build PIP Package
You can build the pip package with either Bazel or make.

With bazel:
```bash
  ./configure.sh
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
```

With Makefile:
```bash
  make zero_out_pip_pkg
```

### Install and Test PIP Package
Once the pip package has been built, you can install it with,
```bash
pip install artifacts/*.whl
```
Then test out the pip package
```bash
cd ..
python -c "import tensorflow as tf;import tensorflow_zero_out;print(tensorflow_zero_out.zero_out([[1,2], [3,4]]))"
```
And you should see the op zeroed out all input elements except the first one:
```bash
[[1 0]
 [0 0]]
```

## Create and Distribute Custom Ops
Now you are ready to write and distribute your own ops. The example in this repository has done the boiling plate work for setting up build systems and package files needed for creating a pip package. We recommend using this repository as a template. 


### Template Overview
First let's go through a quick overview of the folder structure of this template repository.
```
├── gpu  # Set up crosstool and CUDA libraries for Nvidia GPU, only needed for GPU ops
│   ├── crosstool/
│   ├── cuda/
│   ├── BUILD
│   └── cuda_configure.bzl
|
├── tensorflow_zero_out  # A CPU only op
│   ├── cc
│   │   ├── kernels  # op kernel implementation
│   │   │   └── zero_out_kernels.cc
│   │   └── ops  # op interface definition
│   │       └── zero_out_ops.cc
│   ├── python
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── zero_out_ops.py   # Load and extend the ops in python
│   │   │   └── zero_out_ops_test.py  # tests for ops
│   │   └── __init__.py
|   |
│   ├── BUILD  # BUILD file for all op targets
│   └── __init__.py  # top level __init__ file that imports the custom op
│
├── tensorflow_time_two  # A GPU op
│   ├── cc
│   │   ├── kernels  # op kernel implementation
│   │   │   |── time_two.h
│   │   │   |── time_two_kernels.cc
│   │   │   └── time_two_kernels.cu.cc  # GPU kernel
│   │   └── ops  # op interface definition
│   │       └── time_two_ops.cc
│   ├── python
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── time_two_ops.py   # Load and extend the ops in python
│   │   │   └── time_two_ops_test.py  # tests for ops
│   │   └── __init__.py
|   |
│   ├── BUILD  # BUILD file for all op targets
│   └── __init__.py  # top level __init__ file that imports the custom op
|
├── tf  # Set up TensorFlow pip package as external dependency for Bazel
│   ├── BUILD
│   ├── BUILD.tpl
│   └── tf_configure.bzl
|
├── BUILD  # top level Bazel BUILD file that contains pip package build target
├── build_pip_pkg.sh  # script to build pip package for Bazel and Makefile
├── configure.sh  # script to install TensorFlow and setup action_env for Bazel
├── LICENSE
├── Makefile  # Makefile for building shared library and pip package
├── setup.py  # file for creating pip package
├── MANIFEST.in  # files for creating pip package
├── README.md
└── WORKSPACE  # Used by Bazel to specify tensorflow pip package as an external dependency

```
The op implementation, including both c++ and python code, goes under `tensorflow_zero_out` dir for CPU only ops, or `tensorflow_time_two` dir for GPU ops. You will want to replace either directory with the corresponding content of your own ops. `tf` folder contains the code for setting up TensorFlow pip package as an external dependency for Bazel only. You shouldn't need to change the content of this folder. You also don't need this folder if you are using other build systems, such as Makefile. The `gpu` folder contains the code for setting up CUDA libraries and toolchain. You only need the `gpu` folder if you are writing a GPU op and using bazel. To build a pip package for your op, you will also need to update a few files at the top level of the template, for example, `setup.py`, `MANIFEST.in` and `build_pip_pkg.sh`.

### Setup
First, clone this template repo.
```bash
git clone https://github.com/tensorflow/custom-op.git my_op
cd my_op
```

#### Docker
Next, set up a Docker container using the provided Docker image for building and testing the ops. We provide two sets of Docker images for different versions of pip packages. If the pip package you are building against was released before Aug 1, 2019 and has manylinux1 tag, please use Docker images `tensorflow/tensorflow:custom-op-ubuntu14` and `tensorflow/tensorflow:custom-op-gpu-ubuntu14`, which are based on Ubuntu 14.04. Otherwise, for the newer manylinux2010 packages, please use Docker images `tensorflow/tensorflow:custom-op-ubuntu16` and `tensorflow/tensorflow:custom-op-gpu-ubuntu16` instead. All Docker images come with Bazel pre-installed, as well as the corresponding toolchain used for building the released TensorFlow pacakges. We have seen many cases where dependency version differences and ABI incompatibilities cause the custom op extension users build to not work properly with TensorFlow's released pip packages. Therefore, it is *highly recommended* to use the provided Docker image to build your custom op. To get the CPU Docker image, run one of the following command based on which pip package you are building against:
```bash
# For pip packages labeled manylinux1
docker pull tensorflow/tensorflow:custom-op-ubuntu14

# For manylinux2010
docker pull tensorflow/tensorflow:custom-op-ubuntu16
```

For GPU, run 
```bash
# For pip packages labeled manylinux1
docker pull tensorflow/tensorflow:custom-op-gpu-ubuntu14

# For manylinux2010
docker pull tensorflow/tensorflow:custom-op-gpu-ubuntu16
```

You might want to use Docker volumes to map a `work_dir` from host to the container, so that you can edit files on the host, and build with the latest changes in the Docker container. To do so, run the following for CPU
```bash
# For pip packages labeled manylinux1
docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-ubuntu14

# For manylinux2010
docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-ubuntu16
```

For GPU, you want to use `nvidia-docker`:
```bash
# For pip packages labeled manylinux1
docker run --runtime=nvidia --privileged  -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-gpu-ubuntu14

# For manylinux2010
docker run --runtime=nvidia --privileged  -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-gpu-ubuntu16

```

#### Run configure.sh
Last step before starting implementing the ops, you want to set up the build environment. The custom ops will need to depend on TensorFlow headers and shared library libtensorflow_framework.so, which are distributed with TensorFlow official pip package. If you would like to use Bazel to build your ops, you might also want to set a few action_envs so that Bazel can find the installed TensorFlow. We provide a `configure` script that does these for you. Simply run `./configure.sh` in the docker container and you are good to go.


### Add Op Implementation
Now you are ready to implement your op. Following the instructions at [Adding a New Op](https://www.tensorflow.org/extend/adding_an_op), add definition of your op interface under `<your_op>/cc/ops/` and kernel implementation under `<your_op>/cc/kernels/`.


### Build and Test CPU Op

#### Bazel
To build the custom op shared library with Bazel, follow the cc_binary example in [`tensorflow_zero_out/BUILD`](https://github.com/tensorflow/custom-op/blob/master/tensorflow_zero_out/BUILD#L5). You will need to depend on the header files and libtensorflow_framework.so from TensorFlow pip package to build your op. Earlier we mentioned that the template has already setup TensorFlow pip package as an external dependency in `tf` directory, and the pip package is listed as `local_config_tf` in [`WORKSPACE`](https://github.com/tensorflow/custom-op/blob/master/WORKSPACE) file. Your op can depend directly on TensorFlow header files and 'libtensorflow_framework.so' with the following:
```python
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
```

You will need to keep both above dependencies for your op. To build the shared library with Bazel, run the following command in your Docker container
```bash
bazel build tensorflow_zero_out:python/ops/_zero_out_ops.so
```

#### Makefile
To build the custom op shared library with make, follow the example in [`Makefile`](https://github.com/tensorflow/custom-op/blob/master/Makefile) for `_zero_out_ops.so` and run the following command in your Docker container:
```bash
make op
```

#### Extend and Test the Op in Python
Once you have built your custom op shared library, you can follow the example in [`tensorflow_zero_out/python/ops`](https://github.com/tensorflow/custom-op/tree/master/tensorflow_zero_out/python/ops), and instructions [here](https://www.tensorflow.org/extend/adding_an_op#use_the_op_in_python) to create a module in Python for your op. Both guides use TensorFlow API `tf.load_op_library`, which loads the shared library and registers the ops with the TensorFlow framework.
```python
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_zero_out_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_zero_out_ops.so'))
zero_out = _zero_out_ops.zero_out

```

You can also add Python tests like what we have done in `tensorflow_zero_out/python/ops/zero_out_ops_test.py` to check that your op is working as intended.


##### Run Tests with Bazel
To add the python library and tests targets to Bazel, please follow the examples for `py_library` target `tensorflow_zero_out:zero_out_ops_py` and `py_test` target `tensorflow_zero_out:zero_out_ops_py_test` in `tensorflow_zero_out/BUILD` file. To run your test with bazel, do the following in Docker container,

```bash
bazel test tensorflow_zero_out:zero_out_ops_py_test
```

##### Run Tests with Make
To add the test target to make, please follow the example in `Makefile`. To run your python test, simply run the following in Docker container,
```bash
make test_zero_out
```

### Build and Test GPU Op

#### Bazel
To build the custom GPU op shared library with Bazel, follow the cc_binary example in [`tensorflow_time_two/BUILD`](https://github.com/tensorflow/custom-op/blob/master/tensorflow_time_two/BUILD#L29). Similar to CPU custom ops, you can directly depend on TensorFlow header files and 'libtensorflow_framework.so' with the following:
```python
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
```

Additionally, when you ran configure inside the GPU container, `config=cuda` will be set for bazel command, which will also automatically include cuda shared library and cuda headers as part of the dependencies only for GPU version of the op: `if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"])`.

To build the shared library with Bazel, run the following command in your Docker container
```bash
bazel build tensorflow_time_two:python/ops/_time_two_ops.so
```

#### Makefile
To build the custom op shared library with make, follow the example in [`Makefile`](https://github.com/tensorflow/custom-op/blob/master/Makefile) for `_time_two_ops.so` and run the following command in your Docker container:
```bash
make time_two_op
```

#### Extend and Test the Op in Python
Once you have built your custom op shared library, you can follow the example in [`tensorflow_time_two/python/ops`](https://github.com/tensorflow/custom-op/tree/master/tensorflow_time_two/python/ops), and instructions [here](https://www.tensorflow.org/extend/adding_an_op#use_the_op_in_python) to create a module in Python for your op. This part is the same as CPU custom op as shown above.


##### Run Tests with Bazel
Similar to CPU custom op, to run your test with bazel, do the following in Docker container,

```bash
bazel test tensorflow_time_two:time_two_ops_py_test
```

##### Run Tests with Make
To add the test target to make, please follow the example in `Makefile`. To run your python test, simply run the following in Docker container,
```bash
make time_two_test
```




### Build PIP Package
Now your op works, you might want to build a pip package for it so the community can also benefit from your work. This template provides the basic setup needed to build your pip package. First, you will need to update the following top level files based on your op.

- `setup.py` contains information about your package (such as the name and version) as well as which code files to include. 
- `MANIFEST.in` contains the list of additional files you want to include in the source distribution. Here you want to make sure the shared library for your custom op is included in the pip package.
- `build_pip_pkg.sh` creates the package hierarchy, and calls `bdist_wheel` to assemble your pip package.

You can use either Bazel or Makefile to build the pip package.


#### Build with Bazel
You can find the target for pip package in the top level `BUILD` file. Inside the data list of this `build_pip_pkg` target, you want to include the python library target ` //tensorflow_zero_out:zero_out_py` in addition to the top level files. To build the pip package builder, run the following command in Docker container,
```bash
bazel build :build_pip_pkg
```

The bazel build command creates a binary named build_pip_package, which you can use to build the pip package. For example, the following builds your .whl package in the `artifacts` directory:
```bash
bazel-bin/build_pip_pkg artifacts
```

#### Build with make
Building with make also invoke the same `build_pip_pkg.sh` script. You can run,
```bash
make pip_pkg
```

### Test PIP Package
Before publishing your pip package, test your pip package.
```bash
pip install artifacts/*.whl
python -c "import tensorflow as tf;import tensorflow_zero_out;print(tensorflow_zero_out.zero_out([[1,2], [3,4]]))"
```


### Publish PIP Package
Once your pip package has been thoroughly tested, you can distribute your package by uploading your package to the Python Package Index. Please follow the [official instruction](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) from Pypi.


### FAQ

Here are some issues our users have ran into and possible solutions. Feel free to send us a PR to add more entries.


| Issue  |  How to? |
|---|---|
|  Do I need both the toolchain and the docker image? | Yes, you will need both to get the same setup we use to build TensorFlow's official pip package. |
|  How do I also create a manylinux2010 binary? | You can use [auditwheel](https://github.com/pypa/auditwheel) version 2.0.0 or newer.  |
|  What do I do if I get `ValueError: Cannot repair wheel, because required library "libtensorflow_framework.so.1" could not be located` or `ValueError: Cannot repair wheel, because required library "libtensorflow_framework.so.2" could not be located` with auditwheel? | Please see [this related issue](https://github.com/tensorflow/tensorflow/issues/31807).  | -->
