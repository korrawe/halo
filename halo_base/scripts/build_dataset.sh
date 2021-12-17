
build_path_c=../../data/test_data_code/test
# NPROC=2 8 
NPROC=16
# NPROC=0
points_uniform_ratio=0.5
GLOBAL_SCALE=0.4

echo $build_path_c

mkdir -p $build_path_c/pointcloud \
        $build_path_c/points \
        $build_path_c/mesh_scaled \
        $build_path_c/points_iou

python sample_mesh.py $build_path_c/mesh \
    --n_proc $NPROC --resize \
    --global_scale $GLOBAL_SCALE \
    --in_meta_folder $build_path_c/meta \
    --pointcloud_folder $build_path_c/pointcloud \
    --points_folder $build_path_c/points \
    --points_uniform_ratio $points_uniform_ratio \
    --mesh_folder $build_path_c/mesh_scaled \
    --packbits --float16 \
    --overwrite
    # --subfolder

# sample uniform points for IoU calculation
python sample_mesh.py $build_path_c/mesh \
    --n_proc $NPROC --resize \
    --global_scale $GLOBAL_SCALE \
    --in_meta_folder $build_path_c/meta \
    --points_folder $build_path_c/points_iou \
    --packbits --float16 \
    --overwrite
    # --subfolder