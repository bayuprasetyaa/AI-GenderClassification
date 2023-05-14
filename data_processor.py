import os

# Data Manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# Torch library
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from PIL import Image

class GenderDataset():
    '''
    Create dataset and preprocessing.
    
    ## Parameters
    annotation_file (path csv): 
        Path annotation csv file with format (column0 = filename, column1 = label)
    img_dir (path): 
        Image source directory pathtransform (Compose, optional): Image transformation
    target_transform (func, optional): 
        Transformation for target
    batch_size(int, default=None): 
        Batch size
    shuffle(bool, default=True): 
        shuffle the data
    validation(bool, default=True): 
        Create testing set if True
    test_size(int, default=0.2): 
        Proportion for testing set, if validation = True
    random_state(int, default=42): 
        random state
    
    ## Returns
    dataset (dict(keys:values))
        keys: labels of partition (train_set/test_set)
        values: Dataloader object
        
    ## Example
    >> transformer = torchvision.transforms.Compose([
                            transforms.centerCrop(240)])
    
    >> dataset = GenderDataset('gender_classification', 'Dataset/Images', transformer).load()
    
    >> print(dataset)
    
    output
    
    dataset{'train_set':DataloaderObject(), 'test_set':DataloaderObject}
    
    '''
    
    def __init__(self,
                 annotation_file:str,
                 img_dir:str,
                 transform: torchvision.transforms.Compose = None,
                 target_transform: any = None,
                 batch_size:int = 64,
                 shuffle: bool = True,
                 validation: bool = True,
                 test_size: int = 0.2,
                 random_state: int = 42) -> None:
        
        self.annotation_file = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.dataset = {'train':annotation_file}
        self.data_loader = {}
        self.test_set = []
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation = validation
        self.test_size = test_size
        self.random_state = random_state
        
    def load(self):
        if self.validation:
            self.dataset['train'], self.dataset['test'] = train_test_split(self.annotation_file, 
                                                                                   test_size=self.test_size, 
                                                                                   stratify=self.annotation_file.iloc[:,1],
                                                                                   random_state=self.random_state)
        
        for category in self.dataset.keys():
            self.data_loader[category] = DataLoader(_ObjectDataset(self.dataset[category], 
                                                                    self.img_dir, 
                                                                    transform=self.transform[category], 
                                                                    target_transform=self.target_transform), 
                                                    batch_size=self.batch_size, 
                                                    shuffle=self.shuffle)
        
        return self.data_loader


class _ObjectDataset(Dataset):   
    def __init__(self,
                 annotation_file: str,
                 img_dir: str,
                 transform: torchvision.transforms.Compose = None,
                 target_transform: any = None) -> None:
        
        self.img_labels = annotation_file
        self.img_dir = os.path.join(img_dir)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = Image.open(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            
            return image, label 
        
        except FileNotFoundError:
            print(f"Image {self.img_labels.iloc[idx, 0]} Not Found!")