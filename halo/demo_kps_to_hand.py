import torch
import numpy as np
import trimesh
import pickle
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ho3d_root = "/media/korrawe/ssd/ho3d/data/HO3D_V2/"

import sys
# sys.path.insert(0, ".")
from models.halo_adapter.adapter import HaloAdapter

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight


def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh


def create_skeleton_mesh(joints):
    mesh_list = []
    # Sphere for joints
    for idx, j in enumerate(joints):
        joint_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.005)
        joint_sphere.vertices += j.detach().cpu().numpy()
        mesh_list.append(joint_sphere)
    
    parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    # Cylinder for bones
    for idx in range(1, 21):
        ed = joints[idx].detach().cpu().numpy()
        st = joints[parent[idx]].detach().cpu().numpy()
        skel = trimesh.creation.cylinder(0.003, segment=(st, ed))
        mesh_list.append(skel)

    skeleton_mesh = concat_meshes(mesh_list)
    return skeleton_mesh


def render_halo(kps, halo_adapter, renderer, out_img_path=None):
    hand_joints = torch.from_numpy(kps)

    halo_mesh = halo_adapter(hand_joints.unsqueeze(0).cuda() * 100.0, joint_order="mano", original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / 100.0

    # render HALO
    camera_t = np.array([0., 0.05, 2.5])
    rend_img = renderer.render(halo_mesh.vertices * 1.0, halo_mesh.faces, camera_t=camera_t)
    
    skeleton_mesh = create_skeleton_mesh(hand_joints)

    skeleton_img = renderer.render(skeleton_mesh.vertices, skeleton_mesh.faces, camera_t=camera_t)
    final = np.hstack([skeleton_img, rend_img])
    # plt.imshow(final)
    if out_img_path is not None:
        plt.imsave(out_img_path, final)
    # plt.show()
    return 


def main():
    # Get renderer
    renderer = Renderer()

    # Load sampled keypoint list (in metre)
    # The sample key points are the interpolation between hand poses in the HO3D dataset, which are not seen during training
    kps_list = pickle.load(open('../resource/kps_seq.pkl', 'rb'))

    # HALO use the same joint ordering as the internal MANO code for transformation matrices, 
    # which is different from the usual MANO output joint ordering.
    # The convertion functions between different joint orderings can be found in "converter.py"

    # Get HALO adapter
    halo_config_file = "../configs/halo_base/yt3d_b16_keypoint_normalized_fix.yaml"
    halo_adapter = HaloAdapter(halo_config_file, device=device, denoiser_pth=None)

    output_mesh_basedir = "../output/"
    rendered_dir = os.path.join(output_mesh_basedir, "render")

    if not os.path.isdir(output_mesh_basedir):
        os.makedirs(output_mesh_basedir)
    os.makedirs(rendered_dir, exist_ok=True)

    print("=> Rendering HALO from keypoints...")
    for idx, kps in enumerate(kps_list):
        out_img_path = os.path.join(rendered_dir, "%03d.png" % (idx))
        render_halo(kps, halo_adapter, renderer, out_img_path=out_img_path)


# Renderer code taken and modified from MeshTransformer (https://github.com/microsoft/MeshTransformer)
# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

class Renderer(object):
    """
    Render mesh using OpenDR for visualization.
    """

    def __init__(self, width=600, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()

    def render(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if isinstance(self.renderer, TexturedRenderer):
            color = [1.,1.,1.]

        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


    def render_vertex_color(self, vertices, faces=None, img=None,
               camera_t=np.zeros([3], dtype=np.float32),
               camera_rot=np.zeros([3], dtype=np.float32),
               camera_center=None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               vertex_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width

        if faces is None:
            faces = self.faces

        if camera_center is None:
            camera_center = np.array([width * 0.5,
                                      height * 0.5])

        self.renderer.camera = ProjectPoints(rt=camera_rot,
                                             t=camera_t,
                                             f=focal_length * np.ones(2),
                                             c=camera_center,
                                             k=np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] -
                      np.mean(vertices, axis=0)[2])
        far = dist + 20

        self.renderer.frustum = {'near': 1.0, 'far': far,
                                 'width': width,
                                 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(
                    img) * np.array(bg_color)

        if vertex_color is None:
            vertex_color = self.colors['light_blue']


        self.renderer.set(v=vertices, f=faces,
                          vc=vertex_color, bgcolor=np.ones(3))
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=self.renderer.v.shape[0],
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.renderer.r


if __name__ == '__main__':
    main()
