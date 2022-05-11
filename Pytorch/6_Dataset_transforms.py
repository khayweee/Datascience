import torch
import torchvision
from transformers import torch_distributed_zero_first

"""
Complete list of buint-in transforms
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransforamtions, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndarray
ToTensor: from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class


Compose Multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

"""
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    """
    Allow pytorch to compute batches and interations
    for epoch training
    """

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform


    def __getitem__(self, index) -> tuple:
        """
        Apply transform when its supplied
        """
        # get item using index
        # i.e. dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        # len(dataset)
        return self.n_samples


# Custom Transform Class
class ToTensor():
    def __call__(self, sample) -> tuple:
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample) -> tuple:
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# Applying first transform
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(labels))

# Applying seocnd transfomr
compose = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=compose)
first_data = dataset[0]
features, labels = first_data
print(type(features))
print(features)
print(type(labels))