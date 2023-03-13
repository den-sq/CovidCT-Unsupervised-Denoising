

import os
import numpy as np
import tifffile as tl
from natsort import natsorted
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from unet import UNet
from empatches import EMPatches
from argparse import ArgumentParser
emp = EMPatches()

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Covid CT noise removal')
    parser.add_argument('-dr', '--data-dir', help='path to noisy dataset', default='/data/datasets/multiloccovnewpat/32bit_reconstructed_datasets/')
    parser.add_argument('-opdr', '--output-dir', help='output path', default='/data/cleaned_images/') 
    parser.add_argument('--mdpt', help='path to saved weights', default='./ckpts/custom-1730/n2n-epoch27-0.00145.pt')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    return parser.parse_args()

params= parse_args()



pathtonoise=params.data_dir
allimages=[pathtonoise+x for x in (natsorted(os.listdir(pathtonoise)))][-3:]
print("Total Images to Denoise:",len(allimages))

if not os.path.exists(params.output_dir):
    os.mkdir(params.output_dir)
    
model = UNet(in_channels=1)
if torch.cuda.is_available():
    if not params.cuda:
        print("please use argument- cuda to run model")
    else:
        print("Using GPU")
        model=model.cuda()
        model.load_state_dict(torch.load(params.mdpt))
        model.eval()

    

class customdataset(Dataset):

    def __init__(self,listofimgs):

      
        self.list_of_images=listofimgs
        self.trans=transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, index):
        noisy_img =self.list_of_images[index]
        noisy_img=self.trans(noisy_img)
        return noisy_img

def loaddata(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False)
def norma(img):
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm

def tensortoimage(tor):
    
    img=tor.cpu()
    img=img.squeeze(0)
    img=img.squeeze(0)
    img=img.numpy()

    return img
            

for image in allimages:
    imagename=image.split('/')[-1]
    print("cleaning image:",imagename)
    img=tl.imread(image)
    fullsize=img.shape
    trims=int((img.shape[0]%256)/2) #reshape to multiple of 256
    img=img[trims:-trims,trims:-trims] #reshape to multiple of 256
    newsize=img.shape
    img=norma(img)
    img_patches, indices = emp.extract_patches(img, patchsize=256, overlap=0.4)
    datar=customdataset(img_patches) 
    evaldata=loaddata(datar)
    denoised_imgs=[]
    for batch_idx, (source) in enumerate(evaldata):
            if params.cuda and torch.cuda.is_available():
                source = source.cuda()
            # Denoise
            denoised_img = model(source).detach()
            denoised_img=tensortoimage(denoised_img)
            denoised_imgs.append(denoised_img)
    merged=emp.merge_patches(denoised_imgs, indices, mode='avg')
    fullimage=np.zeros((fullsize),dtype='float32')
    fullimage[trims:trims+newsize[0],trims:trims+newsize[1]]=merged
    pathtosave=params.output_dir+'cleaned_'+imagename
    print("saving image to",pathtosave)
    tl.imsave(pathtosave,fullimage)
