# CovidCT-Unsupervised-Denoising


This is an Inference-Only  Repository for cleaning Covid tissue Micro CT samples of size 9680x9680
The code works on reconstructed image data and expects the image to be of type float32
The pretrained model (currently Unet) has weights saved as denoiser.pt and can denoise gaussian-poisson mixtures present in the image
The model is not capable of removing ring artifacts at this stage.

Training Procedure:
This is the general training procedure used for training this model.
1. First, an Ansecombe transform was applied to the noisy images with correlated noise (G+P).
2. Since the inter-slice width is negligible, the change in signal between two consecutive slices is approximately 0.
3. However, noise varies between the two slices. Patches from slice[i] were used as input and the corresponding slice[i+1] patch was used target.
4. Over 100 epochs, the unet learns the mean of the observations (true signal) and produces clean image.


Instructions for running the code:

1. Install Conda and create new environment with python>=3.7 
2. Install the requirements mentioned in the requirements.txt
3. Please use GPU for running the model.

Sample code for running on terminal-

python customeval.py \
--data_dir ../data \
--output_dir ../data \
--mdpt ../denoiser.pt \
--cuda
