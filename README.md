# AdvDefense_CSC
Code for [Adversarial Defense by Stratiﬁed Convolutional Sparse Coding](https://arxiv.org/pdf/1812.00037.pdf).

This is an attack-agnostic adversarial defense based on input transformation. After getting adversarial examples from threaten model, we reconstruct adversarial images (optional: and clean images) with convolutional sparse coding to remove adversarial noise. After reconstruction, adversairal examples are projected to a quai-natural space wehere they share close perceptual features and network capturing features. 
![pipeline](https://github.com/GitBoSun/AdvDefense_CSC/blob/master/images/pipeline.pdf)
## Requirements
You need to install [sporco](https://github.com/bwohlberg/sporco) to operate convolutional sparse coding, 
## Defense Process
### 1. Without Clustering 
Here we don't use DAE to find a cluster that input image belongs to. After learning filter basis from natural images, we direcctly use this basis to reconstruct images. 
### 2. With Clustering
We first train a Denosing Autoencoder with (64,64) natural images. Then we split all images to several clusters based on DAE latent features and learn a filter basis for each cluster. For each input image, we first resize it to (64, 64) and feed it to DAE to find the cluster its latent feature belongs to. Then we reconstruct the input with basis of its cluster. 
## Usage
### recons_data.py
Given a folder with images, it reconstruct those images to another folder with the same image names. 
You need to specify te input path, output path and basis path in this python script. 
example: 
```
python recons_data.py 64 8 0.2 32
```
### basis
It saves some pre-learned filter basis from natural images. In this folder, we have basis for cifar, Imagenet(in resolution 224 and 64). In each subfoler, we have basis with different sparse coefficients. 
```64_p8_lm0.2.npy``` means this basis have 64 filters with size (8, 8) and it's learned under the sparse coefficient 0.2. 
## Results
### CIFAR-10
![](https://github.com/GitBoSun/AdvDefense_CSC/blob/master/images/cifar_compare.pdf)
### ImageNet
![](https://github.com/GitBoSun/AdvDefense_CSC/blob/master/images/imagenet_compare.pdf)


