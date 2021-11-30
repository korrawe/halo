import os
import numpy as np
import trimesh

from utils.intersection import intersection_eval
import matplotlib.pyplot as plt

def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal


def eval_grabnet():
    # For GrabNet
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    # mesh_file_0 = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/binoculars/obj_mesh_10.ply'
    # mesh_file_1 = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/binoculars/rh_mesh_gen_10.ply'

    # hand = trimesh.load(mesh_file_0, process=False)
    # hand = seal(hand)
    # object = trimesh.load(mesh_file_1, process=False)

    # eval_tmp_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/eval_temp'

    # vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True, visualize_file=eval_tmp_dir + '/output.off')
    # print(vol)
    # print(mesh_dist)

    # datalist_file = "/home/korrawe/halo_vae/data/ho3d/datalist.txt"
    datalist_file = "/home/korrawe/halo_vae/data/obman_test/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    # GrabNet model results
    # mesh_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/'
    mesh_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/grab_new_objects/'
    # mesh_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/grab_new_objects_ho3d/'
    vol_list = []
    mesh_dist_list = []
    for object_type in object_list:
        print()
        obj_dir = os.path.join(mesh_dir, object_type)
        # vol_list = []
        # mesh_dist_list = []
        n_sample = 5  # for obman # 10  for ho3d #  20 for GRAB
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # Refine
            # hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_%s.ply' % idx)
            # Coarse
            hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_coarse_%s.ply' % idx)
            obj_mesh_filename = os.path.join(obj_dir, 'obj_mesh_%s.ply' % idx)
            print(hand_mesh_filename)
            print(obj_mesh_filename)
            hand = trimesh.load(hand_mesh_filename, process=False)
            hand = seal(hand)
            object = trimesh.load(obj_mesh_filename, process=False)
            # For Obman Object
            trimesh.repair.fix_normals(object)
            # trimesh.repair.fill_holes(object)
            object = trimesh.convex.convex_hull(object)

            vol, mesh_dist = intersection_eval(hand, object, res=0.001)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
            print(vol)
            # import pdb; pdb.set_trace()

    print(" -- ", object_type, " -- ")
    print(" vol cm3: ", np.mean(vol_list))
    print(" inter dist cm: ", np.mean(mesh_dist_list))
    print(" contact ratio: ", np.mean(np.array(mesh_dist_list) < 0))
    vol_clip = np.clip(vol_list, 0, 20)
    plt.figure()
    plt.hist(vol_clip, bins=20, range=(0, 21))
    plt.show()


def eval_halo():
    # For HALO
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    datalist_file = "/home/korrawe/halo_vae/data/obman_test/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])
    # mesh_file_0 = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes/obj0_gt_obj_mesh.obj'
    # mesh_file_1 = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes/obj000_h000.obj'

    # hand = trimesh.load(mesh_file_0, process=False)
    # # hand = seal(hand)
    # object = trimesh.load(mesh_file_1, process=False)

    # # eval_tmp_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/eval_temp'
    # # eval_tmp_dir = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes'
    # eval_tmp_dir = '/home/korrawe/halo_vae/exp/bmc_loss_grab/generation_latest/meshes'

    # vol, mesh_dist = intersection_eval(hand, object, res=0.001, scale=0.01, visualize_flag=True, visualize_file=eval_tmp_dir + '/output.off')
    # print(vol)
    # print(mesh_dist)

    # GrabNet model results
    # mesh_dir = '/home/korrawe/halo_vae/exp/bmc_loss_grab/generation_latest/meshes/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/meshes/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine/generation/meshes/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation_latest_17/meshes/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation_17_2/meshes/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/meshes/'
    mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation_obman/meshes/'

    vol_list = []
    mesh_dist_list = []

    for obj_idx, object_type in enumerate(object_list):
        print()
        obj_dir = mesh_dir  # os.path.join(mesh_dir, object_type)
        # obj_mesh_filename = os.path.join(obj_dir, 'obj%s_gt_obj_mesh.obj' % obj_idx)
        obj_mesh_filename = os.path.join(obj_dir, '%s_gt_obj_mesh.obj' % object_type)

        # vol_list = []
        # mesh_dist_list = []
        n_sample = 5  # 20
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            # Initial prediction
            hand_mesh_filename = os.path.join(obj_dir, '%s_h%03d.obj' % (object_type, idx))
            # Refine
            # hand_mesh_filename = os.path.join(obj_dir, '%s_h%03d_refine.obj' % (object_type, idx))

            print(hand_mesh_filename)
            print(obj_mesh_filename)
            hand = trimesh.load(hand_mesh_filename, process=False)
            # hand = seal(hand)
            object = trimesh.load(obj_mesh_filename, process=False)
            trimesh.repair.fix_normals(object)
            # trimesh.repair.fill_holes(object)
            object = trimesh.convex.convex_hull(object)
            # object.export(os.path.join(obj_dir, '%s_gt_obj_mesh_repair.obj' % object_type))
            # import pdb; pdb.set_trace()

            vol, mesh_dist = intersection_eval(hand, object, res=0.001, scale=0.01)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
            print(vol)

        print(" -- ", object_type, " -- ")
    print(" vol cm3: ", np.mean(vol_list))
    print(" inter dist cm: ", np.mean(mesh_dist_list))
    print(" contact ratio: ", np.mean(np.array(mesh_dist_list) < 0))

    vol_clip = np.clip(vol_list, 0, 20)
    plt.figure()
    plt.hist(vol_clip, bins=20, range=(0, 21))
    plt.show()
        # import pdb; pdb.set_trace()


def eval_halo_mano():
    # For HALO
    object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    mesh_dir = '/home/korrawe/halo_vae/exp/grab_baseline_mano/generation/meshes/'

    for obj_idx, object_type in enumerate(object_list):
        print()
        obj_dir = mesh_dir  # os.path.join(mesh_dir, object_type)
        # obj_mesh_filename = os.path.join(obj_dir, 'obj%s_gt_obj_mesh.obj' % obj_idx)
        obj_mesh_filename = os.path.join(obj_dir, '%s_gt_obj_mesh.obj' % object_type)

        vol_list = []
        mesh_dist_list = []
        n_sample = 20
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            # Initial prediction
            hand_mesh_filename = os.path.join(obj_dir, '%s_h%03d.obj' % (object_type, idx))
            # Refine
            # hand_mesh_filename = os.path.join(obj_dir, '%s_h%03d_refine.obj' % (object_type, idx))

            print(hand_mesh_filename)
            print(obj_mesh_filename)
            hand = trimesh.load(hand_mesh_filename, process=False)
            hand = seal(hand)
            object = trimesh.load(obj_mesh_filename, process=False)

            vol, mesh_dist = intersection_eval(hand, object, res=0.001, scale=0.01)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
            print(vol)

        print(" -- ", object_type, " -- ")
        print(" vol cm3: ", np.mean(vol_list))
        print(" inter dist cm: ", np.mean(mesh_dist_list))
        print(" contact ratio: ", np.mean(np.array(mesh_dist_list) < 0))

        vol_clip = np.clip(vol_list, 0, 20)
        plt.figure()
        plt.hist(vol_clip, bins=20, range=(0, 21))
        plt.show()
        # import pdb; pdb.set_trace()


def eval_gf():
    # For HALO
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    # datalist_file = "/home/korrawe/halo_vae/data/obman_test/datalist.txt"
    datalist_file = "/home/korrawe/halo_vae/data/ho3d/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])
    # mesh_file_0 = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes/obj0_gt_obj_mesh.obj'
    # mesh_file_1 = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes/obj000_h000.obj'

    # hand = trimesh.load(mesh_file_0, process=False)
    # # hand = seal(hand)
    # object = trimesh.load(mesh_file_1, process=False)

    # # eval_tmp_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/test_grasp_results/eval_temp'
    # # eval_tmp_dir = '/home/korrawe/halo_vae/exp/bmc_loss/generation_latest_grab_number/meshes'
    # eval_tmp_dir = '/home/korrawe/halo_vae/exp/bmc_loss_grab/generation_latest/meshes'

    # vol, mesh_dist = intersection_eval(hand, object, res=0.001, scale=0.01, visualize_flag=True, visualize_file=eval_tmp_dir + '/output.off')
    # print(vol)
    # print(mesh_dist)

    # GrabNet model results
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation_obman/meshes/'
    mesh_dir = '/home/korrawe/GF/grasping_field/output/mano/'
    obj_dir = '/home/korrawe/GF/grasping_field/input/'


    vol_list = []
    mesh_dist_list = []

    for obj_idx, object_type in enumerate(object_list):
        print()
        # obj_dir = mesh_dir  # os.path.join(mesh_dir, object_type)
        # obj_mesh_filename = os.path.join(obj_dir, 'obj%s_gt_obj_mesh.obj' % obj_idx)
        obj_mesh_filename = os.path.join(obj_dir, '%s.obj' % object_type)

        # import pdb; pdb.set_trace()

        # vol_list = []
        # mesh_dist_list = []
        n_sample = 10  # for ho3d # 5 for obman  # 20
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            # Initial prediction
            hand_mesh_filename = os.path.join(mesh_dir, '%s_%d_hand_mano.ply' % (object_type, idx))

            print(hand_mesh_filename)
            print(obj_mesh_filename)
            hand = trimesh.load(hand_mesh_filename, process=False)
            hand = seal(hand)
            object = trimesh.load(obj_mesh_filename, process=False)
            trimesh.repair.fix_normals(object)
            # trimesh.repair.fill_holes(object)
            object = trimesh.convex.convex_hull(object)
            # object.export(os.path.join(obj_dir, '%s_gt_obj_mesh_repair.obj' % object_type))
            # import pdb; pdb.set_trace()

            vol, mesh_dist = intersection_eval(hand, object, res=0.001, visualize_flag=True)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
            print(vol)
            # import pdb; pdb.set_trace()

        print(" -- ", object_type, " -- ")
    print(" vol cm3: ", np.mean(vol_list))
    print(" inter dist cm: ", np.mean(mesh_dist_list))
    print(" contact ratio: ", np.mean(np.array(mesh_dist_list) < 0))

    vol_clip = np.clip(vol_list, 0, 20)
    plt.figure()
    plt.hist(vol_clip, bins=20, range=(0, 21))
    plt.show()


def eval_gf_ho3d_subsample():
    datalist_file = "/home/korrawe/halo_vae/data/ho3d/datalist_500.txt"
    datalist_file_30 = "/home/korrawe/halo_vae/data/ho3d/datalist_30.txt"

    object_list = []
    object_dict = {}
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip())

    listof30 = []
    for obj in object_list:
        seq_name = obj.split('_')[0]
        obj_only = ''.join(i for i in seq_name if not i.isdigit())
        if obj_only in object_dict:
            continue
        object_dict[obj_only] = 1
        listof30.append(obj)

    import pdb; pdb.set_trace()
    # listof30 = np.random.choice(object_list, 30)
    # listof30 = list(listof30)
    listof30.sort()

    with open(datalist_file_30, 'w') as f:
        for fname in listof30:
            f.write(fname + '\n')

    obj_dir = "/media/korrawe/data/works/gf/ho3d_fix/ho3d_correction/ho3d_fixed_simulation/obj/"
    mesh_dir = "/media/korrawe/data/works/gf/ho3d_fix/ho3d_correction/ho3d_fixed_simulation/obj/"

    copy_target_dir = "/home/korrawe/halo_vae/data/ho3d"

    vol_list = []
    mesh_dist_list = []

    # for obj_idx, object_type in enumerate(object_list):
    for obj_idx, object_type in enumerate(listof30):
        print(object_type)
        obj_mesh_filename = os.path.join(obj_dir, '%s_obj.obj' % object_type)
        hand_mesh_filename = os.path.join(mesh_dir, '%s_hand.obj' % (object_type))
        print()
        # import pdb; pdb.set_trace()

        print(hand_mesh_filename)
        print(obj_mesh_filename)
        hand = trimesh.load(hand_mesh_filename, process=False)
        # hand = seal(hand)
        object = trimesh.load(obj_mesh_filename, process=False)
        object.vertices *= [-1., 1., 1.]
        object.vertices = object.vertices - np.mean(object.vertices, 0)
        object.export(os.path.join(copy_target_dir, '%s.obj' % object_type))

    return

def copy_ho3d():
    datalist_file = "/home/korrawe/halo_vae/data/ho3d/datalist.txt"
    object_list = []
    object_dict = {}
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    mesh_dir = "/media/korrawe/data/works/gf/ho3d_fix/ho3d_correction/ho3d_fixed_simulation/obj/"
    copy_target_dir = "/home/korrawe/GF/grasping_field/input_ho3d/"

    for obj_idx, object_type in enumerate(object_list):
        print(object_type)
        obj_mesh_filename = os.path.join(mesh_dir, '%s_obj.obj' % object_type)
        print()
        print(obj_mesh_filename)
        object = trimesh.load(obj_mesh_filename, process=False)
        object.vertices *= [-1., 1., 1.]
        object.export(os.path.join(copy_target_dir, '%s.obj' % object_type))


def eval_ho3d():
    datalist_file = "/home/korrawe/halo_vae/data/ho3d/datalist.txt"

    object_list = []
    object_dict = {}
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    # obj_dir = "/home/korrawe/halo_vae/data/ho3d/"
    mesh_dir = "/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation_ho3d/meshes/"

    vol_list = []
    mesh_dist_list = []

    for obj_idx, object_type in enumerate(object_list):
        print()
        obj_dir = mesh_dir  # os.path.join(mesh_dir, object_type)
        obj_mesh_filename = os.path.join(obj_dir, '%s_gt_obj_mesh.obj' % object_type)

        # import pdb; pdb.set_trace()

        # vol_list = []
        # mesh_dist_list = []
        n_sample = 10  # 20
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            hand_mesh_filename = os.path.join(obj_dir, '%s_h%03d.obj' % (object_type, idx))

            print(hand_mesh_filename)
            print(obj_mesh_filename)
            hand = trimesh.load(hand_mesh_filename, process=False)
            hand = seal(hand)
            object = trimesh.load(obj_mesh_filename, process=False)
            trimesh.repair.fix_normals(object)
            # trimesh.repair.fill_holes(object)
            # object = trimesh.convex.convex_hull(object)
            # object.export(os.path.join(obj_dir, '%s_gt_obj_mesh_repair.obj' % object_type))
            # import pdb; pdb.set_trace()

            vol, mesh_dist = intersection_eval(hand, object, res=0.001, scale=0.01)
            vol_list.append(vol)
            mesh_dist_list.append(mesh_dist)
            print(vol)

        print(" -- ", object_type, " -- ")
    print(" vol cm3: ", np.mean(vol_list))
    print(" inter dist cm: ", np.mean(mesh_dist_list))
    print(" contact ratio: ", np.mean(np.array(mesh_dist_list) < 0))

    vol_clip = np.clip(vol_list, 0, 20)
    plt.figure()
    plt.hist(vol_clip, bins=20, range=(0, 21))
    plt.show()

def main():
    eval_grabnet()
    # eval_halo()
    # eval_halo_mano()
    # eval_gf()
    # eval_gf_ho3d()
    # eval_ho3d()
    # copy_ho3d()


if __name__ == "__main__":
    main()
