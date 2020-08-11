# CvxNet: Differentiable Convex Decomposition (CVPR 2020)

\[[Project Page](http://cvxnet.github.io)\]
\[[Video](https://www.youtube.com/watch?v=Rgi63tT670w)\]
\[[Paper](https://arxiv.org/abs/1909.05736)\]

<div align="center">
  <img width="95%" alt="CvxNet Illustration" src="http://services.google.com/fh/files/misc/cropped_teaser.gif">
</div>

This project provides the open source implementation with pretrained models used
for experiments in the paper. Our work proposes a differentiable way of
representating any shape as the union of convex polytopes. CvxNet can be trained
on large-scale shape collections as an implicit representation while used as an
explicit representation during inference time. Please refer to our paper for
more details.

## Installation

We highly recommend using
[conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
or [virtualenv](https://virtualenv.pypa.io/en/latest/) to run this project. The
code is tested on python 3.7.6 and we recommend using python 3.7+ and install
all the dependencies by:

```bash
git clone https://github.com/tensorflow/graphics.git
cd graphics/tensorflow_graphics/projects/cvxnet
pip install -r requirements.txt
```

Meanwhile, make sure the path to the root directory of tensorflow_graphics is in
your `PYTHONPATH`. If not, please:

```bash
export PYTHONPATH="/PATH/TO/TENSORFLOW_GRAPHICS:$PYTHONPATH"
```

For efficiently extracting smooth mesh surfaces from indicator functions, we
also adapt a tool `libmise` from
[Occupancy Networks](https://github.com/autonomousvision/occupancy_networks). If
you use this tool, please consider citing
[their paper](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks)
as well. To install this extension module written in cython, assuming that we
are still at `graphics/tensorflow_graphics/projects/cvxnet`, run:

```bash
python ./setup.py build_ext --inplace
```

## Prepare the dataset

The model is trained on [ShapeNet](https://www.shapenet.org/). In our
experiments, we use point samples with occupancy labels based on meshes from
this dataset. We also use color renders from
[3D-R2N2](https://github.com/chrischoy/3D-R2N2). In this project, we provide a
small sample dataset which you can use to train and evaluate the model directly.
If you want to use the whole dataset (which might be huge), please visit their
websites ([ShapeNet](https://www.shapenet.org/) and
[3D-R2N2](https://github.com/chrischoy/3D-R2N2)) and download the original data.
We will soon release our point samples, depth renders, and the script to
assembling raw data. To prepare the sample dataset, run:

```bash
wget cvxnet.github.io/data/sample_dataset.zip
unzip sample_dataset.zip -d /tmp/cvxnet
rm sample_dataset.zip
```

and the sample dataset will reside in `/tmp/cvxnet/sample_dataset`.

## Training

To start the training, make sure you have the dataset ready following the above
step and run:

```bash
python train.py --train_dir=/tmp/cvxnet/models/rgb --data_dir=/tmp/cvxnet/sample_dataset --image_input
```

The training record will be saved in `/tmp/cvxnet` so we can launch a
tensorboard by `tensorboard --logdir /tmp/cvxnet` to monitor the training. Also
note that this launches the RGB-to-3D experiment with the same parameters in the
paper. If you want to launch the {Depth}-to-3D experiment, use:

```bash
python train.py --train_dir=/tmp/cvxnet/models/depth --data_dir=/tmp/cvxnet/sample_dataset --n_half_planes=50
```

## Evaluation

First, let's assume that our model for RGB-to-3D is saved in
`/tmp/cvxnet/models/rgb` and the model for {Depth}-to-3D is saved in
`/tmp/cvxnet/models/depth`. We can launch the evaluation job for the RGB-to-3D
model by:

```bash
python eval.py --train_dir=/tmp/cvxnet/models/rgb --data_dir=/tmp/cvxnet/sample_dataset--image_input
```

This will write the IoU (Intersection of Union) on the test set to tensorboard.
Similarly, {Depth}-to-3D model can be evaluated by:

```bash
python eval.py --train_dir=/tmp/cvxnet/models/depth --data_dir=/tmp/cvxnet/sample_dataset --n_half_planes=50
```

We will soon release our pretrained models for RGB-to-3D and {Depth}-to-3D
tasks. If you want to extract meshes as well, please also set `--extract_mesh`
which will automatically save meshes to the `train_dir`. Note that we use a
transformed version of ShapeNet for our point sampling so for full shapenet
evaluation you need to download the transformations which we will release soon.

## Reference

If you find this work useful, please consider citing:

```
@article{deng2020cvxnet,
  title = {CvxNet: Learnable Convex Decomposition},
  author = {Deng, Boyang and Genova, Kyle and Yazdani, Soroosh and Bouaziz, Sofien and Hinton, Geoffrey and Tagliasacchi, Andrea},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
