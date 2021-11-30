
# build_path_c=/home/korrawe/nasa/data/sample_hands_2std/test
# build_path_c=/home/korrawe/nasa/data/FHB/test/Subject_2_squeeze_paper_1/test
# build_path_c=/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test
# build_path_c=/home/korrawe/nasa/data/youtube_hand/resample_PC/train
# build_path_c=/home/korrawe/nasa/data/interhand/test
# build_path_c=/home/korrawe/nasa/data/youtube_hand/ori_vert_fixshape/train
# build_path_c=/home/korrawe/nasa/data/youtube_hand/youtube_keypoint_fixed_normalized/train
build_path_c=/home/korrawe/nasa/data/youtube_hand/youtube_keypoint_no_scale/train
# build_path_c=/home/korrawe/nasa/data/youtube_hand/seq_for_vid_4/test
# build_path_c=/home/korrawe/nasa/data/youtube/shape/617_to_238/test eval_fist
# build_path_c=/home/korrawe/nasa/data/eval_fist/test
# build_path_c=/home/korrawe/nasa/data/youtube_hand/no_rot/val
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


# python sample_mesh.py /home/korrawe/nasa/data/interhand/train/mesh \
#     --n_proc 1 --resize \
#     --global_scale 0.4 \
#     --in_meta_folder /home/korrawe/nasa/data/interhand/train/meta \
#     --pointcloud_folder /home/korrawe/nasa/data/interhand/train/pointcloud \
#     --points_folder /home/korrawe/nasa/data/interhand/train/points \
#     --points_uniform_ratio 0.5 \
#     --mesh_folder /home/korrawe/nasa/data/interhand/train/mesh_scaled \
#     --packbits --float16 \
#     --overwrite