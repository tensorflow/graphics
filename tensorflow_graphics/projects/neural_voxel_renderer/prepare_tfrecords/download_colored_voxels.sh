#!/bin/bash
for i in {0..99}
do
   fname=$(printf "%05d" $i)
   wget -P $1 https://storage.googleapis.com/tensorflow-graphics/notebooks/neural_voxel_renderer/colored_voxels/test-$fname-of-00100.tfrecord
done