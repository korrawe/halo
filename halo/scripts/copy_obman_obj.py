import os
import json
import pdb
import trimesh
import shutil
import numpy as np


split_file = '/media/korrawe/Elements/MPI/obman/split/test_1500.json'
mesh_folder = '/media/korrawe/Elements/MPI/obman/transformed/test/'
out_folder = '/home/korrawe/halo_vae/data/obman_test/'
gf_object_folder = '/home/korrawe/GF/grasping_field/input/'

with open(split_file, 'r') as f:
    obj_list = json.load(f)

# pdb.set_trace()
obj_list = [str(a) for a in obj_list['test'].keys()]

out_file_list = []
max_obj = 30
for i in range(max_obj):
    obj_name = obj_list[i]
    obj_path = os.path.join(mesh_folder, obj_name, 'obj', 'models', 'model_normalized.obj')
    obj_mesh = trimesh.load(obj_path, process=False)

    hand_path = os.path.join(mesh_folder, obj_name, 'hand', 'models', 'model_normalized_sealed.obj')
    hand_mesh = trimesh.load(hand_path, process=False)

    hand_center = hand_mesh.vertices[-1]
    obj_mesh.vertices = obj_mesh.vertices - hand_center
    # pdb.set_trace()

    # object for GF
    gf_outpath = os.path.join(gf_object_folder, obj_name + '.obj')
    obj_mesh.export(gf_outpath)

    # object for halo
    min_p = obj_mesh.vertices.min(0)
    max_p = obj_mesh.vertices.max(0)
    obj_center = (max_p + min_p) / 2.0
    obj_mesh.vertices = obj_mesh.vertices - obj_center
    # obj_mesh.vertices = obj_mesh.vertices * 100.0

    out_file_list.append(obj_name + '.obj')
    outpath_halo = os.path.join(out_folder, obj_name + '.obj')
    obj_mesh.export(outpath_halo)

    # break


with open(os.path.join(out_folder, 'datalist.txt'), 'w') as f:
    for name in out_file_list:
        print(name + '\n')
        f.write(name + '\n')

data = {"filenames": out_file_list}
with open(os.path.join(gf_object_folder, 'input.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
