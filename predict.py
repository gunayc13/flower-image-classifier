#importing necessary libraries
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
import PIL
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict
import time
import argparse
import json

#  Arguments of the script
parser = argparse.ArgumentParser()

parser.add_argument ('image_dir', help = 'Mandatory argument_Image', type = str)
parser.add_argument ('load_dir', help = 'Mandatory argument_Checkpoint', type = str)
parser.add_argument ('--top_k', type = int, default=1)
parser.add_argument ('--category_names', type = str, default=None)
parser.add_argument ('--GPU', type = str)



# Loading the checkpoint
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)        
    model = getattr(torchvision.models, checkpoint['network'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model = load_checkpoint('checkpoint.pth')

def process_image(image):
      
    # Process a PIL image for use in a PyTorch model

    test_image = PIL.Image.open(image)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

# Prediction
def predict(image_path, model, topk, device):
        
    image = process_image (image_path)    

    #converting to tensor
    if device == 'cuda':
        image = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy (image).type (torch.FloatTensor)
   
    image = image.unsqueeze(0)

    #enabling GPU/CPU
    model.to (device)
    image.to (device)

    with torch.no_grad ():
        output = model.forward (image)        
        top_prob, top_labels = torch.topk (output, topk)
        top_prob = top_prob.exp()
    
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob.numpy()[0], mapped_classes


args = parser.parse_args()
file_path = args.image_dir

#device definition
if args.GPU == 'GPU':
    device = 'cuda:0'
else:
    device = 'cpu:0'

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = load_checkpoint (args.load_dir)

if args.top_k:
    class_name = args.top_k
else:
    class_name = 1

# calculating probabilities and classes
probs, classes = predict (file_path, model, class_name, device)

# preparing class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes]

for l in range (class_name):
     print("Number: {}/{}.. ".format(l+1, class_name),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )