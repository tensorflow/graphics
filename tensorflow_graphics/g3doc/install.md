# Installing TensorFlow Graphics

## Stable builds

TensorFlow Graphics depends on [TensorFlow](https://www.tensorflow.org/install)
1.13.1 or above. Nightly builds of TensorFlow (tf-nightly) are also supported.

To install the latest CPU version from
[PyPI](https://pypi.org/project/tensorflow-graphics/), run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

and to install the latest GPU version, run:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

For additional installation help, guidance installing prerequisites, and
(optionally) setting up virtual environments, see the
[TensorFlow installation guide](https://www.tensorflow.org/install).

## Installing from source - macOS/Linux

You can also install from source by executing the following commands:

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## Installing optional packages - Linux

To use the TensorFlow Graphics EXR data loader, OpenEXR needs to be installed.
This can be done by running the following commands:

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```
