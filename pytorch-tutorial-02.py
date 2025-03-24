
### Tutorial 2: Datasets and DataLoaders

# IMPORTANT NOTE: If you get the following error:
#       import matplotlib.pyplot as plt
#       ModuleNotFoundError: No module named 'matplotlib'
# and you know you've already installed it, hit ctrl + shift + p and select the Python interpreter that matches your Python version
# even if the correct interpreter is already selected, just select it again - this should fix the issue
# Update: IF you happen to have solved this issue, successfully run the program, and then woken up the next day, tested the program to make sure it still ran, and found that it DOES NOT and the solution you used last night no longer works... maybe switch to electrical engineering like your dad wanted you to.
# If you don't want to do that, try running this script in PyCharm.

# PyTorch provides two data primitives for handing data samples: torch.utils.data.DatLoader and torch.utils.data.Dataset
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the sample.
# Both allow you to use pre-loaded datasets, or your own data

# PyTorch provides a number of pre-loaded datasets that subclass torch.utils.data.Dataset, and implement functions specific to the particular data.
# You can find them here:
    # Image Datasets: https://pytorch.org/vision/stable/datasets.html
    # Text Datasets: https://pytorch.org/text/stable/datasets.html
    # Audio Datasets: https://pytorch.org/audio/stable/datasets.html

### Loading a Dataset
# Exmpample using the FashionMNIST dataset from TorchVision, a dataset of Zalando's article images consisting of 60,000 training examples and 10,000 test examples
# each example comprises a 28x28 grayscale image and an associated label from one of 10 classes

# Load the dataset with the following parameters:
    # 'root' is the path where the train/test data is stored
    # 'train' specifies training or test dataset
    # 'download = True' downloads the data from the internet if it's not available at root
    # 'transform' and 'target_transform' specify the feature and label transformations

# imports for tutorial part 1
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# imports for tutorial pt 2
import os
import pandas as pd
from torchvision.io import read_image

# import for tutorial part 3
from torch.utils.data import DataLoader

# note the transform = ToTensor()
# this turns our image into a tensor, and scales the RBG values down to 0.0 - 1.0 (by dividing the values by 255)
# the shape of the tensor is (channel,height,width)

# see more about the dataset used here: https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Iterating and Visualizing the Dataset
# we can index Datasets manually like a list: training_date[index]. We use matplotlib to visualize some smaples in our training data
# below, we create a dictionary from our data by assigning indices to the category of clothing item

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# this is a matplot.pyplot method, .figure()
# this methods cerates a new figure or activates an existing figure
# figure() returns a figure object and takes the following paramters, * = optional:
    # *num: int, string (or Figure or Subfigure) which is a unique identifier for the figure
    # figsize: (float, float), default (6.4, 4.8), gives the width, height in inches of the figure
    # dpi: default = 100.0, this is the resolution of the figure (dots-per-inch) but be careful bc it will override the figsize paramters to accomplish target dpi and distort frame
    # facecolor: 'color', default = 'white', this is the background color
    # edgecolor: 'color', default = 'white', the border color (use with linewidth = n)
    # frameon: boolean, default = True, when False, suppresses drawing the figure frame
    # *FigureClass: subclass of Figure, optional usage of custom Figure instance
    # clear: boolean, default = False
    # tight_layout: boolean or dictionary, default = False, when False it will use subplotpars, when True w/o dict, will add default padding to height, width; with dict, will use dict as w/h/rect padding
    # constrained_layout: boolean, default = False, when True will use constrained layout to adjust positioning of plot elements
    # * **kwargs, optional

figure = plt.figure(figsize=(8, 8), facecolor='red', edgecolor='blue', linewidth=5)
cols, rows = 3, 3

# what this for loop does:
# we will construct a grid of  3 x 3 (3 cols, 3 rows defined above) and for each "item" in the grid
# we assign a random integer value to the sample's index, pulling randomly from our data
# then we assign its place in the figure using add_subplot
# this means that for each time we run this figure, we get a new set of images on our grid
# we are randomly selecting the image labels, so sometimes we get multiple of the same kind of clothing item
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

# this line will generate our figure when we run the code
plt.show()


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# data is not being read from the csv, but it is also not causing an error
# the tutorial did not provide the .csv for the training data, and I had to find my own - it may be an issue with the csv that I found online
my_data = CustomImageDataset('mnist_fashion_test.csv', '/images')


### Preparing your data for training with DataLoaders
# the Dataset retrieves our dataset's features and labels one sample at a time.
# While training a model, we typically want to pass samples in "minibatches", reshuffle the data at every epoch (reduces overfitting) and use Python's 'multiprocessing' to speed up data retrieval

# DataLoader is an iterable that abstracts this complexity for us in an easy API

# in machine learning, an "epoch" is one complete pass of the entire training dataset through the entire algorithm
# after one epoch, the model will be able to learn from the data and adjust parameters like weights and biases

# the DataLoader does all the hard work for us (for now)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
# the DataLoader allows us to iterate through the data set as-needed
# each iteration below returns a batch of train_features and train_labels, containing batch_size = 64 features and labels, respectively
# Since we specified shuffle = True, after all batches are iterated over, the data is shuffled

# Display image and label
# first, we created our train_features, train_labels items using next(iter())
# next(iter()) grabs the "next" iteration of the training data, so we can see the images and labels being generated
# the for-loop allows us to see this done several times to inspect several different batches
for i in range(5):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze() # squeeze() removes dimensions of size 1, which is helpful for cleaning up data - you can see in our output, our images do include a dimension of size 1
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
