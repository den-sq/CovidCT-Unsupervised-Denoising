# CTML

This is a python package for ML approaches for handling Micro-CT Data.  Functionality will be expanded over time.

Current functionality is limited to a denoising approach for reconstructed data using an unsupervised (inference-only) model.

### Inference Model Training
#### Procedure:	
1. First, an Ansecombe transform is applied to the noisy images with correlated noise (G+P).
2. Since the inter-slice width is negligible, the change in signal between two consecutive slices is approximately 0.
3. However, noise varies between the two slices. Patches from slice[i] were used as input and the corresponding slice[i+1] patch was used as target.
4. Over iterated epochs, the unet learns the mean of the observations (true signal) and produces a clean image.
	
A pretrained weight set (trained on 9680x9680 sample, 100 epochs) is present at data/denoiser.pt.

#### Usage:
python ctml --data-dir data/train --weights data/final.pt --cuda utraining --nb-epochs 100

For full paramter listing see
python ctml --help
python ctml utraining --help

### Model Application

The finalized model is applied to the input file in patches, that are merged together.

#### Usage:
python ctml --data-dir data/train --weights data/denoiser.pt --cuda udenoise --output-dir=data/clean

For full paramter listing see
python ctml --help
python ctml udenoise --help

(Note that the batch-size parameter listed in python ctml --help is ignored here due to limitations of empatches).

### Current Limits:
1. Data is assumed to be float32.
2. Model cannot remove ring artifacts. 
3. While CUDA is technically not required, model is too computationally intensive for straight CPU usage.

### Installation:
1. Create a conda environment, which is necessary foor use of CUDA GPU tools.
2. Install Pytorch.  Links below as per time of writing.
	1. CUDA >= 11.7: [Current Version](https://pytorch.org/get-started/locally/)
	2. CUDA < 11.7: [Old Version](https://pytorch.org/get-started/previous-versions/)

	As this will limit available python versions and is hardware dependent, should be done before installing other requirements.
3. Install requirements via conda or pip:  
	conda env update -f ctml.yaml  
	pip install -r requirements.txt

	Note that pytorch is required but not listed in the requirements due to variance in CUDA version requirements.