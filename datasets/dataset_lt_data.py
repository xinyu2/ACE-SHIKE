import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
import pickle

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def get_per_class_num(self):
        num_classes = len(np.unique(self.targets))
        cls_num_list = [0] * num_classes
        for label in self.targets:
            cls_num_list[label] += 1
        return cls_num_list

class LT_box_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, path_to_box=None):
        self.img_path = []
        self.labels = []
        self.filename = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                self.filename.append(line.split()[0])
        self.targets = self.labels # Sampler needs to use targets
        self.dict_box={}
        if path_to_box:
            pickle_in = open(path_to_box, 'rb')
            data = pickle.load(pickle_in)
            for box in data:
                for z in box:
                    self.dict_box[z[0]] = z[1]
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        f = self.filename[index]
        tmp_box = self.dict_box[f]
        if tmp_box.shape[0] == 0:
            box = np.zeros(4,dtype=np.float32)
        else:
            area = []
            for b_ in tmp_box:
                b=b_.numpy()
                area.append(b[2]*b[3])
            pick = np.argsort(np.array(area))[0]
            box = tmp_box[pick].numpy()
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, box

    def get_per_class_num(self):
        num_classes = len(np.unique(self.targets))
        cls_num_list = [0] * num_classes
        for label in self.targets:
            cls_num_list[label] += 1
        return cls_num_list