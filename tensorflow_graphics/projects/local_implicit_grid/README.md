## Local Implicit Grid Representations for 3D Scenes

By: [Chiyu "Max" Jiang](http://maxjiang.ml/),
[Avneesh Sud](https://research.google/people/105052/),
[Ameesh Makadia](http://www.ameeshmakadia.com/index.html),
[Jingwei Huang](http://stanford.edu/~jingweih/),
[Matthias Niessner](http://niessnerlab.org/members/matthias_niessner/profile.html),
[Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)

\[[Project Website](http://maxjiang.ml/proj/lig)\] \[[Paper PDF Preprint](https://arxiv.org/abs/2003.08981)\]

![teaser](https://storage.googleapis.com/local-implicit-grids/lig_teaser.gif)

### Introduction

This repository is based on our CVPR 2020 paper:
[Local Implicit Grid Representations for 3D Scenes](https://arxiv.org/abs/2003.08981).
The [project webpage](http://maxjiang.ml/proj/lig) presents an overview of the
project.

Shape priors learned from data are commonly used to reconstruct 3D objects from
partial or noisy data. Yet no such shape priors are available for indoor scenes,
since typical 3D autoencoders cannot handle their scale, complexity, or
diversity. In this paper, we introduce Local Implicit Grid Representations, a
new 3D shape representation designed for scalability and generality. The
motivating idea is that most 3D surfaces share geometric details at some
scale -- i.e., at a scale smaller than an entire object and larger than a small
patch. We train an autoencoder to learn an embedding of local crops of 3D shapes
at that size. Then, we use the decoder as a component in a shape optimization
that solves for a set of latent codes on a regular grid of overlapping crops
such that an interpolation of the decoded local shapes matches a partial or
noisy observation. We demonstrate the value of this proposed approach for 3D
surface reconstruction from sparse point observations, showing significantly
better results than alternative approaches.

Our deep learning code base is written using [Tensorflow](https://www.tensorflow.org/).

### Getting started

Code is tested with python 3.7+ and tensorflow 1.14+. Please install the
necessary dependencies. `pip` is a recommended way to do this.

```bash
pip install -r requirements.txt
```

### Scene reconstruction using pretrained part encoding
Currently we are releasing the evaluation code to use our pretrained model for
scene reconstruction, along with definitions for the local implicit grid layer
and part-autoencoder model. To directly use our script for surface
reconstruction, prepare the input point cloud as a `.ply` file with vertex
attributes: `x, y, z, nx, ny, nz`. See `resample_geometry.py` for creating an
input `.ply` file from a mesh. For demo input data, refer to the inputs
under `demo_data/`.

To reconstruct a meshed surface given an input point cloud,
run `reconstruct_geometry.py` as follows:

```bash
# Be sure to add root of tensorflow_graphics direectory to your PYTHONPATH
# Assuming PWD=<path/to/teensorflow_graphics>
export PYTHONPATH="$PWD:$PYTHONPATH"
pushd tensorflow_graphics/projects/local_implicit_grid/

# using one GPU is sufficient
export CUDA_VISIBLE_DEVICES=0

# download the model weights.
wget https://storage.googleapis.com/local-implicit-grids/pretrained_ckpt.zip
unzip pretrained_ckpt.zip; rm pretrained_ckpt.zip

# fetch a test object and compute point cloud.
mkdir -p demo_data
wget https://cs.uwaterloo.ca/~c2batty/bunny_watertight.obj
mv bunny_watertight.obj demo_data

# reconstruct an object. since objects are much smaller than entire scenes,
# we can use a smaller point number and number of optimization steps to speed
# up.
python reconstruct_geometry.py \
--input_ply demo_data/bunny.ply \
--part_size=0.20 --npoints=2048 --steps=3001

# download more demo data for scene reconstruction.
wget http://storage.googleapis.com/local-implicit-grids/demo_data.zip
unzip demo_data.zip; rm demo_data.zip

# reconstruct a dense scene
python reconstruct_geometry.py \
--input_ply demo_data/living_room_33_1000_per_m2.ply \
--part_size=0.25

# reconstruct a sparser scene using a larger part size
python reconstruct_geometry.py \
--input_ply demo_data/living_room_33_100_per_m2.ply \
--part_size=0.50
```

The part size parameter controls the granularity of the local implicit grid. For
scenes it should be in the range of 0.25 - 0.5 (meters). For objects, it depends
on the scale of the coordinates. Generally for normalized objects (max bounding
box length ~ 1) use a part size of ~0.2. Generally `part_size` should not be
greater than 1/4 of the minimum bounding box width.

### References
If you find our code or paper useful, please consider citing

    @inproceedings{Local_Implicit_Grid_CVPR20,
      title = {Local Implicit Grid Representations for 3D Scenes},
      author = {Chiyu Max Jiang and Avneesh Sud and Ameesh Makadia and Jingwei Huang and Matthias Nießner and Thomas Funkhouser},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }

### Contact

Please contact [Max Jiang](mailto:maxjiang93@gmail.com) or
[Avneesh Sud](mailto:avneesh@google.com) if you have further questions!
