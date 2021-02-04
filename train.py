#importing necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = 'training script parser')

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg16 can be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 1024', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

#setting values data loading
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#device definition cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#data loading, transformation definition
if data_dir:
    train_trans = transforms.Compose ([transforms.RandomRotation (30),
                                       transforms.RandomResizedCrop (224),
                                       transforms.RandomHorizontalFlip (),
                                       transforms.ToTensor (),
                                       transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ])

    valid_trans = transforms.Compose ([transforms.Resize (255),
                                       transforms.CenterCrop (224),
                                       transforms.ToTensor (),
                                       transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ])

    test_trans = transforms.Compose ([transforms.Resize (255),
                                      transforms.CenterCrop (224),
                                      transforms.ToTensor (),
                                      transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_trans)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_trans)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_trans)

    
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)
    

#label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):
    if arch == 'vgg16': 
        model = models.vgg16 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.5)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.5)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: 
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.5)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.5)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: 
        arch = 'vgg13'        
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:            
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.5)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.5)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: 
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.5)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.5)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    model.classifier = classifier 
    return model, arch

# validation function definition
def validation(model, valid_loader, criterion):
    model.to (device)

    valid_loss = 0
    accuracy = 0
    
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# model loading
model, arch = load_model (args.arch, args.hidden_units)

# training and initialization
criterion = nn.NLLLoss ()
if args.lrn: 
    
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to (device) 

if args.epochs:
    epochs = args.epochs
else:
    epochs = 1

print_every = 30
steps = 0

for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (train_loader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad ()        
        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item ()

        if steps % print_every == 0:
            model.eval () 
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0
            model.train()

model.to ('cpu') 
model.class_to_idx = train_image_datasets.class_to_idx 

#creating dictionary
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
#saving trained model
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')