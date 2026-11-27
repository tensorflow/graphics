source dataset_shapenet/config.sh

# Function for processing a single model
reorganize_choy2016() {
  modelname=$(basename -- $5)
  output_path="$4/$modelname"
  build_path=$3
  choy_vox_path=$2
  choy_img_path=$1

  points_file="$build_path/4_points/$modelname.npz"
  points_out_file="$output_path/points.npz"

  pointcloud_file="$build_path/4_pointcloud/$modelname.npz"
  pointcloud_out_file="$output_path/pointcloud.npz"

  vox_file="$choy_vox_path/$modelname/model.binvox"
  vox_out_file="$output_path/model.binvox"

  img_dir="$choy_img_path/$modelname/rendering"
  img_out_dir="$output_path/img_choy2016"

  metadata_file="$choy_img_path/$modelname/rendering/rendering_metadata.txt"
  camera_out_file="$output_path/img_choy2016/cameras.npz"

  echo "Copying model $output_path"
  mkdir -p $output_path $img_out_dir

  cp $points_file $points_out_file
  cp $pointcloud_file $pointcloud_out_file
  cp $vox_file $vox_out_file

  python dataset_shapenet/get_r2n2_cameras.py $metadata_file $camera_out_file
  counter=0
  for f in $img_dir/*.png; do
    outname="$(printf '%03d.jpg' $counter)"
    echo $f
    echo "$img_out_dir/$outname"
    convert "$f" -background white -alpha remove "$img_out_dir/$outname"
    counter=$(($counter+1))
  done
}

export -f reorganize_choy2016

# Make output directories
mkdir -p $OUTPUT_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  CHOY2016_IMG_PATH_C="$CHOY2016_PATH/ShapeNetRendering/$c"
  CHOY2016_VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"
  mkdir -p $OUTPUT_PATH_C

  ls $CHOY2016_VOX_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    reorganize_choy2016 $CHOY2016_IMG_PATH_C $CHOY2016_VOX_PATH_C \
      $BUILD_PATH_C $OUTPUT_PATH_C {}

  echo "Creating split"
  python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.2
done
