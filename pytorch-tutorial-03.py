

### Transforms
# We use transforms to perform some manipulation of the data and make it suitable for training.
# All TorchVision datasets have two parameters:
    # transform to modify the features
    # target_transform to modify the labels
# These parameters accept callables containing the transformation logic.
# The torchvision.transforms module offers several commonly-used transforms out of the box.

# The FashionMNIST features are in PIL Image format, with integers as labels
# For training, we need the features as TENSORS and the labels as "one-hot encoded" tensors
    # one-hot encoding is used to convert categorical data into binary format by creating separate columns for each label
    # all features that are part of one category - in this case we'll say "shoes" - gets a 1 in that column, and a 0 in every other column

# for these transformations, we use ToTensor and Lambda

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# lambda functions in Python are small, anonymous (not bound to a name) functions, often used as an argument to other functions
# target_transform = a Lambda function that adds a tensor of 0's (our one-hot encoding) with the label for that category equal to 1
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


