 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision.models import googlenet, GoogLeNet_Weights

from callback import Callback, Checkpoint
from train import loop_fn
from data_preprocess import GenderDataset


# Config
BATCH_SIZE = 64
CROP_SIZE = 224
MODEL = 'GoogleNet-Base'

config = {
"batch_size": BATCH_SIZE,
"crop_size": CROP_SIZE
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

# Data Preprocessing

IMAGE_PATH = 'Datasets\Images'
LABEL_FILE = 'Datasets\gender_classification.csv'

transformer = {'train_set':transforms.Compose([
    transforms.Resize(230),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(CROP_SIZE, scale= (.8, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
               'test_set': transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
}


dataset = GenderDataset(LABEL_FILE,IMAGE_PATH, transform=transformer, batch_size=BATCH_SIZE).load()
print(dataset)
trainLoader = dataset['train_set']
testLoader = dataset['test_set']

# Model
model = googlenet()
model.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout2d(.1),

            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout2d(.1),

            nn.Linear(256, 2),
            nn.LogSoftmax()
        )

model.to(device)

# Set up Training Phase
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters())
callback = Callback(model, device, config, outdir = "GoogleNet_Base_64_8020") 

# Training Model
while True:
    CM = 0
    
    train_cost, train_score = loop_fn("train", trainLoader, model, criterion, optimizer, device, focus_on = 'f1')
    
    with torch.no_grad():
        test_cost, test_score, CM_test = loop_fn("test", testLoader, model, criterion, optimizer, device, focus_on = 'f1')
        CM += CM_test

    ## LOGGING
    callback.log(train_cost, test_cost, train_score, test_score)

    ## Checkpoint
    callback.save_checkpoint()

    ## Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()

    ## EARLY STOPPING
    if callback.early_stopping(model, monitor='test_score'):
        callback.plot_cost()
        callback.plot_score()
        break

    print(90 * "-")
    print()
    print()