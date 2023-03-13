# CovidCT-Unsupervised-Denoising


This is an Inference-Only  Repository for cleaning Covid tissue Micro CT samples of size 9680x9680
The code works on reconstructed image data and expects the image to be of type float32
The pretrained model (currently Unet) has weights saved as noise-remover.pt and can denoise gaussian-poisson mixtures present in the image
The model is not capable of removing ring artifacts at this stage.


Instructions for running the code:

1. Install Conda and create new environment with python>=3.7 
2. Install the requirements mentioned in the requirements.txt
