# Neural Radiance Fields in Tensorflow-Graphics

Neural Radiance Fields [1] became a milestone in the field of neural rendering and novel view synthesis. In this repository, we provide a re-implementation of the original paper and we illustrate how the TF-Graphics functionalities can be extended to different NeRF-based methods.

## Installation

We tested our implementation using python3.9 and tensorflow 2.6.0.

1) [Install TF-Graphics](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/install.md)

2) Dependencies
  - numpy
  - Pillow
  - abseil

Example installation:

```
pip install --upgrade tensorflow-graphics
pip install numpy Pillow absl-py
```

## Colab
For a small training and testing demo, see this [colab](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/radiance_fields/TFG_tiny_nerf.ipynb)

## Data preparation
Please download the data from the original [repository](https://github.com/bmild/nerf). In this tutorial we experimented with the synthetic data (lego, ship, boat, etc) that can be found [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (link to the [1] authors storage).

```
DATASET_DIR=/path/to/nerf/data/
CHECKPOINT_DIR=/path/to/checkpoints/
OUTPUT_DIR=/path/to/output/
```

## Training
```
python nerf/train.py --dataset_dir $DATASET_DIR --dataset_name lego --checkpoint_dir $CHECKPOINT_DIR
```

## Testing
```
python nerf/eval.py --dataset_dir $DATASET_DIR --dataset_name lego --checkpoint_dir $CHECKPOINT_DIR --output_dir $OUTPUT_DIR
```

The default parameters for both training and testing are set for the synthetic datasets.

#### References
[1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng, NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV2020
