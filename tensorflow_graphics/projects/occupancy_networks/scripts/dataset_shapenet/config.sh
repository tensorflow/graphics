ROOT=..

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/data/external/ShapeNetCore.v1
CHOY2016_PATH=$ROOT/data/external/Choy2016
BUILD_PATH=$ROOT/data/ShapeNet.build
OUTPUT_PATH=$ROOT/data/ShapeNet

NPROC=12
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
03001627
02958343
04256520
02691156
03636649
04401088
04530566
03691459
02933112
04379243
03211117
02828884
04090263
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
