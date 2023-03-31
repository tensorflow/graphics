source dataset_shapenet/config.sh
# Make output directories
mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c

  mkdir -p $build_path_c/0_in \
           $build_path_c/1_scaled \
           $build_path_c/1_transform \
           $build_path_c/2_depth \
           $build_path_c/2_watertight \
           $build_path_c/4_points \
           $build_path_c/4_pointcloud \
           $build_path_c/4_watertight_scaled \

  echo "Converting meshes to OFF"
  lsfilter $input_path_c $build_path_c/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
     meshlabserver -i $input_path_c/{}/model.obj -o $build_path_c/0_in/{}.off;
  
  echo "Scaling meshes"
  python $MESHFUSION_PATH/1_scale.py \
    --n_proc $NPROC \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
  
  echo "Create depths maps"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=render --n_proc $NPROC \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth
  
  echo "Produce watertight meshes"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=fuse --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/2_watertight \
    --t_dir $build_path_c/1_transform

  echo "Process watertight meshes"
  python sample_mesh.py $build_path_c/2_watertight \
      --n_proc $NPROC --resize \
      --bbox_in_folder $build_path_c/0_in \
      --pointcloud_folder $build_path_c/4_pointcloud \
      --points_folder $build_path_c/4_points \
      --mesh_folder $build_path_c/4_watertight_scaled \
      --packbits --float16
done
