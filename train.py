#importing libraries
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from collections import OrderedDict
import PIL
from PIL import Image
import argparse
import time
import json
from torch.autograd import Variable



# Initiate variables with default values
arch = 'vgg16'
hidden_units = 5120
learning_rate = 0.001
epochs = 1
device = 'cpu'

# Parser
parser = argparse.ArgumentParser()

parser.add_argument ('data_dir', help = 'Mandatory argument -- Data Directory', type = str)
parser.add_argument ('--save_dir', help = 'Optional argument -- Save Directory', type = str)
parser.add_argument ('--arch', help = 'Vgg16 can be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 1024', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--gpu',action='store_true', help = "Option to use GPU")

#data loading
args = parser.parse_args ()

# Select parameters entered in command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Get the num of classes from the directory
number_train_classes = len(os.listdir(train_dir))
number_valid_classes = len(os.listdir(valid_dir))
number_test_classes = len(os.listdir(test_dir))

if (number_train_classes != number_valid_classes) or (number_train_classes != number_test_classes) or (number_valid_classes != number_test_classes):
    print('Error: number of train, valid test classes is not the same')
    
number_classes = number_train_classes

train_trans = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_trans = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
test_trans = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataSet = datasets.ImageFolder(train_dir, transform=train_trans)
valid_dataSet = datasets.ImageFolder(valid_dir, transform=valid_trans)
test_dataSet = datasets.ImageFolder(test_dir, transform=test_trans)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataSet, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=64, shuffle=True)


#label mapping
#import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    #print(cat_to_name)


# Build and train your network
model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Updating classifier

dropout_probability = 0.5
in_features = 25088
out_features = 1024

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(1024, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier


# Train the network - Define deep learning method
# gpu section -- use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train the classifier parameters
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# change to device
model.to(device);

epochs = 5
validation_step = True


# Train the classifier layers using backpropogation using the pre-trained network to get features
print('\n*** Training process started! ***')
start_training_time = time.time()

for epoch in range(epochs):
    train_loss = 0
    for inputs, labels in train_loader:     
        
        # Move input and label tensors to the default device
        inputs = inputs.to(device)
        labels = labels.to(device)        
        optimizer.zero_grad()        
        log_probabilities = model.forward(inputs)
        loss = criterion(log_probabilities, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print('\nEpoch: {}/{} '.format(epoch + 1, epochs),
          '\nTraining:\t\t\tLoss: {:.4f}  '.format(train_loss / len(train_loader))
         )
           
    if validation_step == True:
        
        valid_loss = 0
        valid_accuracy = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)                
                log_probabilities = model.forward(inputs)
                loss = criterion(log_probabilities, labels)        
                valid_loss = valid_loss + loss.item()
        
                # Calculate accuracy
                probabilities = torch.exp(log_probabilities)
                top_probability, top_class = probabilities.topk(1, dim = 1)                
                equals = top_class == labels.view(*top_class.shape)                
                valid_accuracy = valid_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
        
        model.train()
       
        print("Validation:\t\tLoss: {:.4f}  ".format(valid_loss / len(valid_loader)),
              "\tAccuracy: {:.4f}".format(valid_accuracy / len(valid_loader)))
        
        
end_training_time = time.time()

print('\n*** Training process completed! ***\n')
      
training_time = end_training_time - start_training_time
print('Training time: {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))

test_loss = 0
test_accuracy = 0
model.eval()


print('*** Validation started! ***')
start_time = time.time()

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    log_probabilities = model.forward(inputs)
    loss = criterion(log_probabilities, labels)

    test_loss = test_loss + loss.item()

    # Calculate accuracy
    probabilities = torch.exp(log_probabilities)
    top_probability, top_class = probabilities.topk(1, dim = 1)

    equals = top_class == labels.view(*top_class.shape)

    test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()

end_time = time.time()
print('*** Validation ended! ***')
validation_time = end_time - start_time
print('Validation time: {:.0f}m {:.0f}s'.format(validation_time / 60, validation_time % 60))

print("\nTest:\t\t\t\tLoss: {:.4f}  ".format(test_loss / len(test_loader)),
      "\tAccuracy: {:.4f}".format(test_accuracy / len(test_loader)))


# training and initialization
criterion = nn.NLLLoss ()
if args.lrn:     
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to ('cpu') 
model.class_to_idx = train_dataSet.class_to_idx 

#Save the checkpoint 
model.class_to_idx = train_dataSet.class_to_idx

checkpoint = {'network': 'vgg16',
              'input_size': in_features,
              'output_size': number_classes,                
              'classifier' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'mapping': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

#creating checkpoint
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }