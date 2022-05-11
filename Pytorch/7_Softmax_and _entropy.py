from audioop import cross
import torch
import torch.nn as nn
import numpy as np


"""
Softmax
"""
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x, dim=0)
print('softmax tensor: ', outputs)

"""
Cross Entropy
"""
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# Y must abe one hot encoded
# if class 0 : [1,0,0]
Y= np.array([0.9,0.1,0])

# y_pred 
y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,y_pred_good)
l2 = cross_entropy(Y,y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'loss2 numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()
# nn.CrossEntropyLoss applies
# nn.logSoftMax and negative log likelihood loss
# -> Soft max is already applied
# Y has class labels and not OneHot encoded
Y = torch.tensor([0])
# n_samples , n_classes = 1x3
y_pred_good = torch.tensor([[2.0,1.0,0.1]]) #Scores
y_pred_bad = torch.tensor([[0.5,2.0,0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1.item())
print(l2.item())

# To get actual prediction
_, prediction1 = torch.max(y_pred_good, dim=1)
print(_)
_, prediction2 = torch.max(y_pred_bad, dim=1)
print(prediction1)
print(prediction2)

# multiple samples
# e.g. 3
print('Multiple Samples #########')
Y = torch.tensor([2,0,1])
y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]])
y_pred_bad = torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]])
print(l1.item())
print(l2.item())

# To get actual prediction
_, prediction1 = torch.max(y_pred_good, dim=1)
print(_)
_, prediction2 = torch.max(y_pred_bad, dim=1)
print(prediction1)
print(prediction2)

# multiple samples