# NASA: Neural Articulated Shape Approximation
## (ECCV 2020)

\[[Project Page](http://nasa-eccv20.github.io)\]
\[[Video](https://twitter.com/taiyasaki/status/1286006705722200064?s=20)\]
\[[Paper](https://arxiv.org/abs/1912.03207)\]

<div align="center">
  <img width="95%" alt="NASA Illustration" src="http://services.google.com/fh/files/misc/teaser.png">
</div>

This project provides the open source implementation of the experiments in the
paper.
NASA is an alternative framework that enables representation of articulated
deformable objects using neural indicator functions that are conditioned on
pose.
Please refer to our paper for more details.

## Installation

We highly recommend using
[conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
or [virtualenv](https://virtualenv.pypa.io/en/latest/) to run this project. The
code is tested on python 3.7.6 and we recommend using python 3.7+ and install
all the dependencies by:

```bash
git clone https://github.com/tensorflow/graphics.git
cd graphics/tensorflow_graphics/projects/nasa
pip install -r requirements.txt
```

Meanwhile, make sure the path to the root directory of tensorflow_graphics is in
your `PYTHONPATH`. If not, please:

```bash
export PYTHONPATH="/PATH/TO/TENSORFLOW_GRAPHICS:$PYTHONPATH"
```

Our project also makes use of the `libmise` module from the `cvxnet` project
which is also within the `tensorflow_graphics` package. Please follow
[the instruction](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/projects/cvxnet/README.md#installation)
to install the module before proceeding to the follow sections.

## Prepare the dataset

The model is trained on [AMASS](https://amass.is.tue.mpg.de/). Please consider
citing the [AMASS paper](https://arxiv.org/abs/1904.03278) if you use this code
base. Here we use a sample dataset of 10 motions from a subject for the
following instructions. To prepare the sample dataset, run:

```bash
wget https://nextcloud.mpi-klsb.mpg.de/index.php/s/7M6PYywGEzkRzPp/download?path=%2F&files=sample_dataset.zip
unzip sample_dataset.zip -d /tmp/nasa
rm sample_dataset.zip
```

and the sample dataset will reside in `/tmp/nasa/sample_dataset`. Note the data
is roughly **10G** in size. Please make sure you have enough space under `/tmp`
or you can specify another location and replace all the following `/tmp/nasa`
with that location.

## Training

To start the training, make sure you have the dataset ready following the above
step and run:

```bash
python train.py \
  --train_dir=/tmp/nasa/models/deform \
  --data_dir=/tmp/nasa/sample_dataset \
  --subject=0 \
  --motion=1
```

This will launch a training job for the *leave-one-out* reconstruction training
of the motion 1. *Leave-one-out* means that we train
on 9 motion sequences of this subject and test on the unseen motion sequence.

The training record will be saved in `/tmp/nasa` so we can launch a tensorboard
by `tensorboard --logdir /tmp/nasa` to monitor the training. Also note that this
job uses the best model proposed in the paper, the **D** (Deform) model. If you
want to train an **R** (Rigid) model, please add the `--nouse_joint` flag.

## Evaluation

First, let's assume that our model is saved in `/tmp/nasa/models/deform`. We
can launch the evaluation job for the **D** model by:

```bash
python eval.py \
  --train_dir=/tmp/nasa/models/deform \
  --data_dir=/tmp/nasa/sample_dataset \
  --subject=0 \
  --motion=1
```

This will write the IoU (Intersection of Union) on the test sequence to
tensorboard. Note that this evaluation job will continuously evaluating new
checkpoints unless there's no new checkpoints in 30 minutes. If you don't want
this behavior, you can kill it once you see the IoU point appears on your
tensorboard.

If you want to extract meshes for the test sequence, please set
`--gen_mesh_only`. It will automatically save meshes to the `meshes` folder
under your `train_dir` and will terminate after generating the whole sequence.
Similar evaluation can be done for the **R** model with
the `--nouse_joint` flag.

## Tracking

We also provide a simple script to showcase the tracking applications shown in
the paper.

Suppose we want to track the motion sequence that we preserved as the
test set. We can simply use the model we just trained and run:

```bash
python track.py \
  --train_dir=/tmp/nasa/models/deform \
  --data_dir=/tmp/nasa/sample_dataset \
  --joint_data=/tmp/nasa/sample_dataset/connectivity.npy \
  --subject=0 \
  --motion=1
```

The output meshes will be save into the `tracked_reparam` folder under your
`train_dir`. The input pointcloud will also be saved to `pointcloud_reparam`.
The IoU of the between the tracked results and ground truth will be saved to
`tracked_reparam/iou.txt`. The tracked motion sequence will look like below:

<img width="28%" alt="NASA Tracking" src="http://services.google.com/fh/files/misc/track.gif"></im>

## Reference

If you find this work useful, please consider citing:

```
@article{deng2020nasa,
  title = {NASA: Neural articulated shape approximation},
  author = {Deng, Boyang and Lewis, JP and Jeruzalski, Timothy and Pons-Moll, Gerard and Hinton, Geoffrey and Norouzi, Mohammad and Tagliasacchi, Andrea},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020}
}
```
