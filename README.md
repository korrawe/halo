# HALO: A Skeleton-Driven Neural Occupancy Representation for Articulated Hands
**Oral Presentation, 3DV 2021**

### Korrawe Karunratanakul, Adrian Spurr, Zicong Fan, Otmar Hilliges, Siyu Tang  <br/>  ETH Zurich

![halo_teaser](/assets/teaser.jpg "HALO teaser")

[![report](https://img.shields.io/badge/Project-Page-blue)](https://korrawe.github.io/HALO/HALO.html)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2109.11399)

Video: [Youtube](http://www.youtube.com/watch?feature=player_embedded&v=QBiAN8Bobuc) <br/>

## Abstract

We present Hand ArticuLated Occupancy (HALO), a novel representation of articulated hands that bridges the advantages of 3D keypoints and neural implicit surfaces and can be used in end-to-end trainable architectures. Unlike existing statistical parametric hand models (e.g.~MANO), HALO directly leverages the 3D joint skeleton as input and produces a neural occupancy volume representing the posed hand surface.
The key benefits of HALO are (1) it is driven by 3D keypoints, which have benefits in terms of accuracy and are easier to learn for neural networks than the latent hand-model parameters; (2) it provides a differentiable volumetric occupancy representation of the posed hand; (3) it can be trained end-to-end, allowing the formulation of losses on the hand surface that benefit the learning of 3D keypoints.
We demonstrate the applicability of HALO to the task of conditional generation of hands that grasp 3D objects. The differentiable nature of HALO is shown to improve the quality of the synthesized hands both in terms of physical plausibility and user preference.

## Updates

- December 1, 2021: Initial release for version 0.01 with demo.

# Running the code
## Dependencies
- [im2mesh library from Occupancy Network](https://github.com/autonomousvision/occupancy_networks) for point sampling and marching cubes
- ```requirements.txt```
- [MANO model](https://github.com/hassony2/manopth) for data preprocessing

The easiest way to run the code is to use [conda](https://docs.conda.io/en/latest/miniconda.html). The code is tested on Ubuntu 18.04.


## Implicit surface from keypoints
![halo_hand](/assets/halo_hand.gif "HALO teaser")
To try a demo which produces an implicit hand surface from the input keypoints, run:
```
cd halo
python demo_kps_to_hand.py
```
The demo will run the marching cubes algorithm and render each image in the animation above sequentially. The output images are in the ```output``` folder. The provided sample sequence are interpolations beetween 17 randomly sampled poses from the unseen HO3D dataset  .


## Dataset
- The HALO-base model is trained using [Youtube3D hand](https://github.com/arielai/youtube_3d_hands) dataset. We only use the hand mesh ground truth without the images and videos. We provide the preprocessed data in the evaluation section.
- The HALO-VAE model is trained and test on the [GRAB](https://grab.is.tue.mpg.de/) dataset

# Evaluation
## HALO base model (implicit hand model)
To generate the mesh given the 3D keypoints and precomputed transformation matrices, run:
```
cd halo_base
python generate.py CONFIG_FILE.yaml
```
To evaluate the hand surface, run:
```
python eval_meshes.py
```
We provide the preprocessed test set of the Youtube3D [here](https://drive.google.com/drive/folders/1dRUPfWvr51VmaTmjHC6xvNrSQmkYMOSH?usp=sharing).
In addition, you can also find the produced meshes from our keypoint model on the same test set [here](https://drive.google.com/drive/folders/1dvz0R39Fk_iX2SnZ4G9tDyzfuOy8sbZN?usp=sharing).


## HALO-VAE
To generate grasps given 3D object mesh, run:
```
python generate.py HALO_VAE_CONFIG_FILE.ymal --test_data DATA_PATH --inference
```
The evaluation code for contact/interpenetration and cluster analysis can be found in ```halo/evaluate.py``` and ```halo/evaluate_cluster.py``` accordningly. The intersection test demo is in ```halo/utils/interscetion.py```

# Training
## HALO base model (implicit hand model)

### Data Preprocessing
Each data point consists of 3D keypoints, transformation matrices, and a hand surface. To speed up the training, all transformation matrices are precomputed, either by out Canonicalization Layer or from the MANO. Please check ```halo/halo_base/prepare_data_from_mano_param_keypoints.py``` for details. After the metadata is processed, sample points and compute occupancy values with ```bash build_dataset.sh```.
We use the surface point sampling and occupancy computation method from the [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks)
 
### Run
To train HALO base model (implicit functions), run:
```
cd halo_base
python train.py
```


## HALO-VAE
To train HALO-VAE, run:
```
cd halo
python train.py
```
HALO_VAE requires a HALO base model trained using the transformation matrices from the Canonicalization Layer. The weights of the base model are not updated during the VAE training.

# BibTex
```
@inproceedings{karunratanakul2021halo,
  title={A Skeleton-Driven Neural Occupancy Representation for Articulated Hands},
  author={Karunratanakul, Korrawe and, Spurr, Adrian and Fan, Zicong and Hilliges, Otmar and Tang, Siyu},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```


## References

Some code in our repo uses snippets of the following repo:

- The occupancy computation and point sampling is based on [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).
- We use [MANO layer](https://github.com/hassony2/manopth) implementation from Yana Hasson
- We use the renderer from [MeshTransformer](https://github.com/microsoft/MeshTransformer) in our demo.

Please consider citing them if you found the code useful.


# Acknowledgement

We  sincerely  acknowledge  Shaofei Wang and Marko Mihajlovic for the insightful discussionsand helps with the baselines.


