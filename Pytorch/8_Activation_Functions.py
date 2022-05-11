"""
To inject non-linearities into Neural networks
- Regular neural networks without Activation Functions
  are just regular linear regressions

Popular Activation Functions
- Step Function (Binary)
    - 1 if x >= 0,
    - 0 o/w
- Sigmoid
    - f(x) = 1/(1+e^(-x))
- TanH (Hyperbolic Tangent Function) \
    - Scaled Sigmoid Function
    - 2/(1+e^(-2x)) - 1
- ReLU
    - Most Popular
    - max(0, x)
- Leaky ReLU
    - tries to solve Vanishing Gradient problem
        - Prevent 0 gradient updates
    - f(x) = x if x >= 0
    - ax o/w, a usually 0.001
- Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1 (Create nn modules)
class NeuralNet(nn.modules):
    """
    Feed Forward Neural Network that outputs 1 value
    """
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: number of input features
        :param hidden_size: number of features in hidden layer
        """
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size,
                                 out_features=hidden_size)
        self.relu = nn.ReLU()
        # nn.Sigmoid()
        # nn.LeakyReLU()
        # nn.TanH()
        self.linear2 = nn.Linear(in_features=hidden_size,
                                 out_features=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        Required Function for nn.module to train
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out

