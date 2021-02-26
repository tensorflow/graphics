# From Points to Multi-Object 3D Reconstruction

[Francis Engelmann](https://francisengelmann.github.io/), [Konstantinos Rematas](http://www.krematas.com/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), [Vittorio Ferrari](https://sites.google.com/corp/view/vittoferrari)
<br>
[Paper](https://arxiv.org/abs/2012.11575)

### Introduction

This directory is based on our CVPR paper [From Points to Multi-Object 3D Reconstruction](https://arxiv.org/abs/2012.11575).

We propose a method to detect and reconstruct multiple 3D objects from a single
RGB image. The key idea is to optimize for detection, alignment and shape
jointly over all objects in the RGB image, while focusing on realistic and
physically plausible reconstructions. To this end, we propose a keypoint
detector that localizes objects as center points and directly predicts all
object properties, including 9-DoF bounding boxes and 3D shapes -- all in a
single forward pass. The proposed method formulates 3D shape reconstruction as
a shape selection problem, i.e. it selects among exemplar shapes from a given
database. This makes it agnostic to shape representations, which enables a
lightweight reconstruction of realistic and visually-pleasing shapes based on
CAD-models, while the training objective is formulated around point clouds and
voxel representations. A collision-loss promotes non-intersecting objects,
further increasing the reconstruction realism. Given the RGB image, the
presented approach performs lightweight reconstruction in a single-stage, it
is real-time capable, fully differentiable and end-to-end trainable.
Our experiments compare multiple approaches for 9-DoF bounding box estimation,
evaluate the novel shape-selection mechanism and compare to recent methods in
terms of 3D bounding box estimation and 3D shape reconstruction quality.

### Getting started
Work in progress: note that this is an initial deployment of the original code,
so some errors may occur.


##### Train CoReNet pairs:
```
train_multi_objects/train.py
--alsologtostderr
--stderrthreshold=info
--logdir='/data/occluded_primitives/logs/n8/'
--tfrecords_dir='/data/occluded_primitives/data_corenet/triplets'
--shapenet_dir='/data/occluded_primitives/shapenet'
--xm_runlocal
--number_hourglasses=1
--num_overfitting_samples=3
--num_classes=14
--max_num_objects=2
--soft_shape_labels=True
--collision_weight=1.0
--run_graph=False
--batch_size=1
--learning_rate=0.0001
--debug=True
--train
```

##### Validate CoReNet triplets
```
train_multi_objects/train.py
--alsologtostderr
--stderrthreshold=info
--logdir='/data/occluded_primitives/multi_objects/logs/18543882/35-pose_pc_pc_weight=10.0,projected_pose_pc_pc_weight=0.1,shapes_weight=0.01,sizes_3d_weight=100.0,soft_shape_labels=True,soft_shape_labels_a=32'
--tfrecords_dir='/data/occluded_primitives/data_corenet/triplets'
--max_num_objects=3
--xm_runlocal
--run_graph=False
--debug=False
--soft_shape_labels=True
--part_id=-2
--val
--eval_only=True
--local_plot_3d=False
--qualitative=True
```


### References
If you find our code or paper useful, please consider citing

    @inproceedings{Engelmann20,
      title = {From Points to Multi-Object 3D Reconstruction},
      author = {Francis Engelmann and Konstantinos Rematas and Bastian Leibe and Vittorio Ferrari},
      booktitle = {arxiv},
      year = {2020}
    }

### Contact

Please contact [Francis Engelmann](mailto:engelmann@vision.rwth-aachen.de)

