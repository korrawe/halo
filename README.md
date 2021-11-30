# HALO: A Skeleton-Driven Neural Occupancy Representation for Articulated Hands
**Oral Presentation, 3DV 2021**

### Korrawe Karunratanakul, Adrian Spurr, Zicong Fan, Otmar Hilliges, Siyu Tang  <br/>  ETH Zurich

![halo_teaser](/assets/teaser.jpg "HALO teaser")

Paper: [arXiv](https://arxiv.org/abs/2109.11399) <br/>
Video: [Youtube](http://www.youtube.com/watch?feature=player_embedded&v=QBiAN8Bobuc) <br/>
Project page: [HALO project page](https://korrawe.github.io/HALO/HALO.html) <br/>



# Abstract
We present Hand ArticuLated Occupancy (HALO), a novel representation of articulated hands that bridges the advantages of 3D keypoints and neural implicit surfaces and can be used in end-to-end trainable architectures. Unlike existing statistical parametric hand models (e.g.~MANO), HALO directly leverages the 3D joint skeleton as input and produces a neural occupancy volume representing the posed hand surface.  
The key benefits of HALO are
(1) it is driven by 3D keypoints, which have benefits in terms of accuracy and are easier to learn for neural networks than the latent hand-model parameters;
(2) it provides a differentiable volumetric occupancy representation of the posed hand;
(3) it can be trained end-to-end, allowing the formulation of losses on the hand surface that benefit the learning of 3D keypoints.  
We demonstrate the applicability of HALO to the task of conditional generation of hands that grasp 3D objects. The differentiable nature of HALO is shown to improve the quality of the synthesized hands both in terms of physical plausibility and user preference. 

# Running the code

## Implicit surface from keypoints
![halo_hand](/assets/halo_hand.gif "HALO teaser")
To try a demo which produces an implicit hand surface from the input keypoints, run:
```
cd halo
python demo_kps_to_hand.py
```
The demo will run the marching cubes algorithm and render each image in the animation above sequentially. The output images are in the ```output``` folder. The provided sample sequence are interpolations beetween 17 randomly sampled poses from the unseen HO3D dataset  .


# Evaluation
## Hand model

## HALO-VAE

# Training
## Hand model
To train HALO base model (implicit functions), run:
```
cd halo_base
python train.py
```
Each data point consists of 3D keypoints, transformation matrices, and a hand surface. To speed up the training, all transformation matrices are precomputed, either by out Canonicalization Layer or from the MANO.
We use the surface point sampling and occupancy computation method from the [Occupancy Networks Code](https://github.com/autonomousvision/occupancy_networks)


## HALO-VAE


# BibTex
```
@inproceedings{karunratanakul2021halo,
  title={A Skeleton-Driven Neural Occupancy Representation for Articulated Hands},
  author={Karunratanakul, Korrawe and, Spurr, Adrian and Fan, Zicong and Hilliges, Otmar and Tang, Siyu},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```

# Acknowledgement
We  sincerely  acknowledge  ShaofeiWang and Marko Mihajlovic for the insightful discussionsand helps with the baselines.
