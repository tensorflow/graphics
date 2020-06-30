# Dataset generation for Neural Voxel Renderer

___

The [training](https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/projects/neural_voxel_renderer/train.ipynb) 
and [inference](https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/projects/neural_voxel_renderer/demo.ipynb)
examples use demo data to illustrate the functionality of Neural Voxel Renderer (NVR).
In this document, we describe how to generate the the full dataset to train NVR from scratch.

**Warning:** the generated TFRecords will take ~350GB of disk space. 

___

## Download the  colored voxels

This dataset contains the colored voxels of 2040 chairs. The size of the dataset is 
**~16GB**. Each shape is represented as 128<sup>3</sup> x 4 voxel grid, where 
each voxel contains an RGB and occupancy value. The color was obtained from a single image 
aligned with the voxels.

``` bash
PATH_TO_COLOR_VOXELS=/tmp/colored_voxels/
mkdir $PATH_TO_COLOR_VOXELS
bash download_colored_voxels.sh $PATH_TO_COLOR_VOXELS
```

## Download the synthetic images

The dataset contains the target images (rendered using Blender) and all the necessary
information that was used to set-up the scene in 3D (object rotation, translation, camera parameters, etc.).
The size of the dataset is **~400MB**

``` bash
PATH_TO_SYNTHETIC_DATASET=/tmp/synthetic_dataset/
mkdir $PATH_TO_SYNTHETIC_DATASET
wget -P $PATH_TO_SYNTHETIC_DATASET https://storage.googleapis.com/tensorflow-graphics/notebooks/neural_voxel_renderer/blender_dataset/default_chairs_test.tfrecord
wget -P $PATH_TO_SYNTHETIC_DATASET https://storage.googleapis.com/tensorflow-graphics/notebooks/neural_voxel_renderer/blender_dataset/default_chairs_train.tfrecord
```

## Run the script

The script iterates over all the synthetic images and pairs them with the corresponding
colored voxels, placed according to the scene set-up. Additionally, it estimates 
the rendered image directly from the voxels which is used as additional input in
NVR plus.

``` python
PATH_TO_TFRECORDS=/tmp/tfrecords/
mkdir $PATH_TO_TFRECORDS
python generate_tfrecords_nvr_plus.py -- --mode test --voxels_dir $PATH_TO_COLOR_VOXELS --images_dir $PATH_TO_SYNTHETIC_DATASET --output_dir $PATH_TO_TFRECORDS
python generate_tfrecords_nvr_plus.py -- --mode train --voxels_dir $PATH_TO_COLOR_VOXELS --images_dir $PATH_TO_SYNTHETIC_DATASET --output_dir $PATH_TO_TFRECORDS
```

