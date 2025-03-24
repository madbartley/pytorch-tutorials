
### Build the Neural Network

# Neural networks comprised of layers/modules that perform operations on data.
# The torch.nn namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module.
# A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.

# this section builds a neural network (nn) to classify images in the FashionMNIST dataset

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# this section will use an accelerator if one is available
# otherwise, we will just use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
print('\n\n')

# Define the class

# we will define our nn by subclassing "nn.Module", and initializing nn layers in __init__
# NOTE about parameters: calling nn.Layer classes will automatically define parameters when we subclass from nn.Module
# for instance, nn.Linear here will automatically name their parameters weight and bias
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # nn.Linear takes 3 arguments: size of the input, size of the output, and (optionally) whether there is a bias argument, but this defaults to True
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# now we create an instance of NeuralNetwork and move it to the "device", then print its structure
model = NeuralNetwork().to(device)
# this will show us the setup of the type of model we are using, as defined in our class
print(model)

# to use the model, we must pass in input data
# this will execute the model's forward() as well as some background operations (the source code of which you can view here: https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)
# NOTE: do NOT call model.forward() directly - this will cause it to bypass some of the inherited functions of nn.Module and you will get errors!

# Recall: a Softmax function converts a vector of K real numbers into a probability distribution of K possible outcomes
# this represents the predictions of each type of possible classifcation

# these next lines of code return a 2D tensor with dim=0 being the output of 10 raw predicted values for each class
# dim=1 corresponds to the individual values of each output
# we then get the prediction probabilities by passing it through an instance of nn.Softmax module
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print('\n\n')

# Model layers
# we will use a sample mini-batch of 3 images of size 28x28
input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.Flatten
# we initiate the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (the minibatch dimnesion at dim=0 is maintained)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear
# The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
# you can read more about the linear layer here: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# after linear transformations, we apply non-linear activations to create complex mappings between the model's inputs and outputs
# this introduces nonlinearity and facilitates nn learning
# note: we are using nn.ReLU here but there are other activations to be used too
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
print(f"After ReLU: {hidden1}")

# nn.Sequential
# an ordered container of modules
# the data gets passed in the same order as defined
# you can use sequential containers to put together a quick network like seq_modules, as follows:
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
# the last linear layer of the nn returns logits, raw values in [-inf,inf] which are passed to nn.Softmax
# the logits are scaled values [0,1] representing the model's predicted probabilities for each class
# the dim parameter indicates the dimension along which the values must sum to 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print('\n\n')

# Model Parameters
# parameterization here refers to the associated weights and biases that are optimized during training
# our nn.Module subclass automatically tracks the fields defined in our model, and we can use parameters() or named_parameters() to view them
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print("Next layer: ", '\n')
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n\n")
