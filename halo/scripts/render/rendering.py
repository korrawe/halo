import numpy as np

import pyrender
import trimesh

class GraspRender():
    def __init__(self, camera_pose = None, perspective = None, light_intensity = None):
        self.scene = pyrender.Scene()

        if perspective is None:
            perspective = np.pi / 4.0
        camera = pyrender.PerspectiveCamera(yfov=perspective, aspectRatio=1.0)
        if camera_pose is None:
            camera_pose = np.array([
                                   [1.0,  0.0, 0.0, 0.0],
                                   [0.0,  1.0, 0.0, 0.0],
                                   [0.0,  0.0, 1.0, 0.5],
                                   [0.0,  0.0, 0.0, 1.0],
                                ])
        self.scene.add(camera, pose=camera_pose)

        if light_intensity is None:
            light_intensity = 1.0  # 5.0
        light = pyrender.SpotLight(color=np.ones(3), intensity=light_intensity,
                               innerConeAngle=np.pi/4.0,
                               outerConeAngle=np.pi/2.0)
        self.scene.add(light, pose=camera_pose)

    def clear_geometry(self):
        all_nodes = self.scene.get_nodes()
        for node in all_nodes:
            if not node.mesh is None:
                self.scene.remove_node(node)

    def add_geometry(self, trimesh_meshes_list):
        for mesh in trimesh_meshes_list:
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            self.scene.add(pyrender_mesh)

    def render(self, resolution = None):
        if resolution is None:
            resolution = 512
        r = pyrender.OffscreenRenderer(resolution, resolution)
        color, depth = r.render(self.scene)
        return color
