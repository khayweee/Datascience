# MNIST
# DataLoader
#   - Load DataSet 
# Transformation
#   - Apply Transformation
# Multilayer Neural Net, Activation Function
# Loss and Optimizer
# Training Loop (Batch training)
# GPU support


from cgi import test
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
input_size = 784 # 28 x 28 image
hidden_size = 100
num_classes = 10 # 10 Different classes in MNIST dataset
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()
# torch.Size([100, 1, 28, 28]) torch.Size([100])
# (batch, channel, length, width), 
print(samples.shape, labels.shape)

for i in range(10):
    # 2rows, 3columns
    plt.subplot(2,5,i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # Softmax would be handled by the crossentropy loss
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        #-> 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward() # backprop
        optimizer.step() # Update parameters
        

        if (i + 1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')