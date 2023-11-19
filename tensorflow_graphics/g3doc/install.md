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

## Installing optional packages

### Linux

To use the TensorFlow Graphics EXR data loader, OpenEXR needs to be installed.
This can be done by running the following commands:

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```

### Windows

Download and install from https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr (Unofficial) with appropriate python version. Say, for Python 3.6.9, 64 Bit Windows 10 OS, install via `pip install OpenEXR-1.3.2-cp36-cp36m-win_amd64.whl`


## Errors and resolutions

- While installing `tensorflow-graphics`, `OpenEXR` may FAIL to get installed. Resolution is to install `OpenEXR` first then re-run `pip install --upgrade tensorflow-graphics`.
- While installing `OpenEXR`, it may error out saying FAILed to find `Imathbox.h`. It is due to missing `libopenexr-dev` . Resolution is to install `OpenEXR` as per OS (mentioned above) and re-install `tensorflow-graphics`.

