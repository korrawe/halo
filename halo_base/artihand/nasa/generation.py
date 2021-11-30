import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
from skimage import measure

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        with_color_labels (bool): whether to assign part-color to the output mesh vertices
        convert_to_canonical (bool): whether to reconstruct mesh in canonical pose (for debugging)
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 with_color_labels=False,
                 convert_to_canonical=False,
                 simplify_nfaces=None,
                 preprocessor=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.with_color_labels = with_color_labels
        self.convert_to_canonical = convert_to_canonical
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.bone_colors = np.array([
            (119, 41, 191, 255), (75, 170, 46, 255), (116, 61, 134, 255), (44, 121, 216, 255), (250, 191, 216, 255), (129, 64, 130, 255),
            (71, 242, 184, 255), (145, 60, 43, 255), (51, 68, 187, 255), (208, 250, 72, 255), (104, 155, 87, 255), (189, 8, 224, 255),
            (193, 172, 145, 255), (72, 93, 70, 255), (28, 203, 124, 255), (131, 207, 80, 255)
            ], dtype=np.uint8
        )

    def generate_mesh(self, data, return_stats=True, threshold=None):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        bone_lengths = data.get('bone_lengths')
        if bone_lengths is not None:
            bone_lengths = bone_lengths.to(device)
        kwargs = {}

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        # print(c.size())

        # z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(c, bone_lengths=bone_lengths, stats_dict=stats_dict, threshold=threshold, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, bone_lengths=None, stats_dict={}, threshold=None, **kwargs):
        ''' Generates mesh from latent.
        Args:
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        if threshold is None:
            threshold = self.threshold

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            # values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            values = self.eval_points(pointsf, c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                # import pdb; pdb.set_trace()
                values = self.eval_points(
                    pointsf, c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        mesh = self.extract_mesh(value_grid, c, bone_lengths=bone_lengths, stats_dict=stats_dict, threshold=threshold)
        return mesh

    def eval_points(self, p, c=None, bone_lengths=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # occ_hat = self.model.decode(pi, z, c, **kwargs).logits
                occ_hat = self.model.decode(pi, c, bone_lengths=bone_lengths,**kwargs)

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    occ_hat = -1 * occ_hat

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat
    
    def eval_point_colors(self, p, c=None, bone_lengths=None):
        ''' Re-evaluates the outputted points from marching cubes for vertex colors.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        pointsf = torch.FloatTensor(p).to(self.device)
        p_split = torch.split(pointsf, self.points_batch_size)
        point_labels = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # occ_hat = self.model.decode(pi, z, c, **kwargs).logits
                # import pdb; pdb.set_trace()
                _, label = self.model.decode(pi, c, bone_lengths=bone_lengths, return_model_indices=True)
                
            point_labels.append(label.squeeze(0).detach().cpu())
            # print("label", label[:40])

        label = torch.cat(point_labels, dim=0)
        label = label.detach().cpu().numpy()
        return label

    def extract_mesh(self, occ_hat, c=None, bone_lengths=None, stats_dict=dict(), threshold=None):
        ''' Extracts the mesh from the predicted occupancy grid.occ_hat
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        if threshold is None:
            threshold = self.threshold

        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # Skimage marching cubes
        # # try:
        # if True:
        #     # value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)
        #     value_grid = occ_hat_padded
        #     verts, faces, normals, unused_var = measure.marching_cubes_lewiner(
        #         value_grid, min(threshold, value_grid.max()))
        #     del normals
        #     verts -= 1
        #     verts /= np.array([
        #         value_grid.shape[0] - 3, value_grid.shape[1] - 3,
        #         value_grid.shape[2] - 3
        #     ],
        #                     dtype=np.float32)
        #     verts = 1.1 * (verts - 0.5)
        #     # verts = scale * (verts - 0.5)
        #     # verts = verts * gt_scale + gt_center
        #     faces = np.stack([faces[..., 1], faces[..., 0], faces[..., 2]], axis=-1)
        #     mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        #     # vertices = verts
        #     return mesh
        # # except:  # pylint: disable=bare-except
        # #     return None

        if vertices.shape[0] == 0:
            mesh = trimesh.Trimesh(vertices, triangles)
            return mesh  #  None

        # Get point colors
        if self.with_color_labels:
            vert_labels = self.eval_point_colors(vertices, c, bone_lengths=bone_lengths)
            vertex_colors = self.bone_colors[vert_labels]

            # Convert the mesh vertice back to canonical pose using the trans matrix of the label
            # self.convert_to_canonical = False # True
            # convert_to_canonical = True
            if self.convert_to_canonical:
                vertices = self.convert_mesh_to_canonical(vertices, c, vert_labels)
                vertices = vertices # * 2.5 * 2.5
        else:
            vertex_colors = None

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            # normals = self.estimate_normals(vertices, z, c)
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               vertex_colors=vertex_colors, ##### add vertex colors
                            #    face_colors=face_colors, ##### try face color
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            # self.refine_mesh(mesh, occ_hat, z, c)
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh
    
    def convert_mesh_to_canonical(self, vertices, trans_mat, vert_labels):
        ''' Converts the mesh vertices back to canonical pose using the input transformation matrices
        and the labels.
        Args:
            vertices (numpy array?): vertices of the mesh
            c (tensor): latent conditioned code c. Must be a transformation matices without projection.
            vert_labels (tensor): labels indicating which sub-model each vertex belongs to.
        '''
        # print(trans_mat.shape)
        # print(vertices.shape)
        # print(type(vertices))
        # print(vert_labels.shape)

        pointsf = torch.FloatTensor(vertices).to(self.device)
        # print("pointssf before", pointsf.shape)
        # [V, 3] -> [V, 4, 1]
        pointsf = torch.cat([pointsf, pointsf.new_ones(pointsf.shape[0], 1)], dim=1)
        pointsf = pointsf.unsqueeze(2)
        # print("pointsf", pointsf.shape)

        vert_trans_mat = trans_mat[0, vert_labels]
        # print(vert_trans_mat.shape)
        new_vertices = torch.matmul(vert_trans_mat, pointsf)

        vertices = new_vertices[:,:3].squeeze(2).detach().cpu().numpy()
        # print("return", vertices.shape)

        return vertices # new_vertices

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        # z, c = z.unsqueeze(0), c.unsqueeze(0)
        c = c.unsqueeze(0)

        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            # occ_hat = self.model.decode(vi, z, c).logits
            occ_hat = self.model.decode(vi, c)
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.
        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                # self.model.decode(face_point.unsqueeze(0), z, c).logits
                self.model.decode(face_point.unsqueeze(0), c)
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh