import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] =  "osmesa" # "egl" #

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
import argparse
import glob
import sys


parser = argparse.ArgumentParser('render mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to directory of input meshes.')
parser.add_argument('--out_folder', type=str,
                    default=None,
                    help='Output images directory.')
parser.add_argument('--rot_file', type=str,
                    help='rotation for good visualization.')
parser.add_argument('--off', action='store_true',
                    help='Input is .off file.')


def render_mesh(meshfile, outfile=None, rotation=None, canonical_pose=False, show=True):
    """
    Render mesh
    Args:
        meshfile (str): Occupancy Network model
        outfile (str): imported yaml config
        rotation (tuple): three rotation radians
        canonical_pose (bool): 
    """
    scene = pyrender.Scene()

    # Load mesh using trimesh and put it in a scene
    mesh_trimesh = trimesh.load(meshfile)
    # mesh_trimesh.faces = np.zeros(0)
    # canonical_pose = True
    canonical_pose = False
    if canonical_pose:
        mesh = pyrender.Mesh.from_points(mesh_trimesh.vertices, colors=mesh_trimesh.visual.vertex_colors)
        rad = np.pi / 180.0 *  -90.0
        scene.add(mesh, pose=np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0,  np.cos(rad), -np.sin(rad), 0.0],
            [0.0,  np.sin(rad),  np.cos(rad), 0.0],
            [0.0,  0.0,  0.0, 1.0],
            ])
        )
        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        s = np.sqrt(2)/2
        camera_pose = np.array([
            [0.0, -s,   s,   0.3],
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  s,   s,   0.45],
            [0.0,  0.0, 0.0, 1.0],
            ])
    else:
        mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

        if rotation:
            rad_x, rad_y, rad_z = rotation
        else:
            # rad_x, rad_y, rad_z = [0.0, 0.0, 0.0]
            rad_x, rad_y, rad_z = [3.0, 3.0, -1.0]

        # scene.add(mesh, pose=np.eye(4))
        # scene.add(mesh, pose=np.array([
        #     [1.0,  0.0,  0.0, 0.0],
        #     [0.0,  0.0, -1.0, 0.0],
        #     [0.0,  1.0,  0.0, 0.0],
        #     [0.0,  0.0,  0.0, 1.0],
        #     ])
        # )
        rot_x = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0,  np.cos(rad_x), -np.sin(rad_x), 0.0],
            [0.0,  np.sin(rad_x),  np.cos(rad_x), 0.0],
            [0.0,  0.0,  0.0, 1.0],
            ]
        )
        rot_y = np.array([
            [np.cos(rad_y) , 0.0, np.sin(rad_y), 0.0],
            [0.0           , 1.0, 0.0, 0.0],
            [-np.sin(rad_y), 0.0, np.cos(rad_y), 0.0],
            [0.0,  0.0,  0.0, 1.0],
            ]
        )
        rot_z = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0.0, 0.0],
            [np.sin(rad_z),  np.cos(rad_z), 0.0, 0.0],
            [0.0          , 0.0           , 1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
            ]
        )
        pose_mat = np.matmul(rot_y, rot_x)
        pose_mat = np.matmul(rot_z, pose_mat)
        scene.add(mesh, pose=pose_mat)

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        s = np.sqrt(2)/2
        # camera_pose = np.array([
        #     [0.0, -s,   s,   0.5],
        #     [1.0,  0.0, 0.0, -0.1],
        #     [0.0,  s,   s,   0.65],
        #     [0.0,  0.0, 0.0, 1.0],
        #     ])
        camera_pose = np.array([
            [0.0, -s,   s,   0.3],
            [1.0,  0.0, 0.0, -0.05],
            [0.0,  s,   s,   0.2],
            [0.0,  0.0, 0.0, 1.0],
            ])
    
    scene.add(camera, pose=camera_pose)
    

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.SpotLight(color=np.ones(3), intensity=2.0,
                                innerConeAngle=np.pi/4.0)  # 16.0)
    camera_pose = np.array([
        [0.0, -s,   s,   0.2],
        [1.0,  0.0, 0.0, -0.05],
        [0.0,  s,   s,   0.45],
        [0.0,  0.0, 0.0, 1.0],
        ])
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(900, 900)
    color, depth = r.render(scene)

    # cam_equal_aspect_3d(ax, verts.numpy())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.subplot(1,2,1)
    ax.axis('off')
    ax.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # if show:
    #     plt.show()

    # plt.savefig('foo.png')
    # plt.savefig('foo.pdf')
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)

def process_path(in_path, args):

    # data = np.load(in_path)
    # # print(data.files)
    # # ['points', 'occupancies', 'loc', 'scale']
    # points, occupancies, loc, scale = data['points'], data['occupancies'], data['loc'], data['scale']
    # joints_trans, verts, vert_labels = data['joints_trans'], data['verts'], data['vert_labels']

    # print(points.shape)
    # print(occupancies.shape)
    # print(loc)
    # print(scale)
    # print("joints_trans", joints_trans.shape)
    # print("verts", verts.shape)
    # print("vert_labels", vert_labels.shape)

    # occupancies = np.unpackbits(occupancies)
    # print(occupancies.shape)
    # print(occupancies)

    # display_points(points, occupancies, joints_trans, verts, vert_labels)
    pass


def main(args):
    if args.off:
        ext = '.off'
        mesh_folder = os.path.join(args.in_folder, 'mesh_scaled')
    else:
        ext = '.obj'
        # mesh_folder = os.path.join(args.in_folder, 'meshes')
        mesh_folder = args.in_folder
    input_files = glob.glob(os.path.join(mesh_folder, '*' + ext))
    # print(input_files)
    # import pdb; pdb.set_trace()

    if not args.out_folder:
        args.out_folder = os.path.join(args.in_folder, 'render')
    if args.out_folder and not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    if args.rot_file:
        with open(args.rot_file, "r") as f:
            lines = f.readlines()
            rotation = [np.pi/180.0 * float(d) for d in lines[1].strip().split(",")]
    else:
        rotation = [0., 0., 0.]

    for p in input_files:
        print(p)
        if args.out_folder:
            obj_name = os.path.splitext(os.path.basename(p))[0]
            out_img = os.path.join(args.out_folder, obj_name + ".png")
        else:
            out_img = None
        print(out_img)
        # process_path(p, args)
        render_mesh(p, out_img, rotation)
        # break


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)