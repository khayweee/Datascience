"""
epoch = 1 one complete forward and backward pass of all training samples
batch_size = number of training samples in one forward and backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples
eg. 100 samples, batch_size = 20 --> 100/20 = 5 iterations per epoch

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

    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]


    def __getitem__(self, index) -> tuple:
        # get item using index
        # i.e. dataset[0]
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        # len(dataset)
        return self.n_samples

dataset = WineDataset()

# Using the Dataset
# first_data = dataset[0]
# features, labels = first_data
# print(features),
# print(labels)

# Using the Dataloader Class
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
dataloader = DataLoader(dataset=dataset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=0 )  #multiple sub processor to make loading faster
# dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print("total_samples, n_iterations")
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for idx, (inputs, labels) in enumerate(dataloader):
        # Forward and Backward and update
        if (idx+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {idx+1}/{n_iterations}, inputs {inputs.shape}')


# Pytorch built in dataset
# MNIST dataset
torchvision.datasets.MNIST()
# fashion-mnist, cifar