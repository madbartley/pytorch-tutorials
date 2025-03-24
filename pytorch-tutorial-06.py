

# Optimizing Model Parameters

# now that we have a model and we want to train it, we need to validate and test it by optimizing its parameters on our data
# in each iteration, the model makes a guess about the output, calculates the error in its guess (loss), collects the derivatives of the error w/r to its parameters, then optimizes them using gradient descent
# you can read more about this process here: https://www.youtube.com/watch?v=tIeHLnjs5U8

### Because this section needs the code from previous models, I'm just copy/pasting the full code here without my notes attached for easier viewing
# the summary here is that we load the FashionMNIST dataset and create the test and train sets
# then, we define our nn.Module-subclassed model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Hyperparameters (Epochs, Batch Size, Learning Rate)
# hp's are adjustable parameters that let you control the model optimization process
# different hp values can impact model training/convergence rates
# you can read more about that here: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

# Brief review of these 3 hps:
    # epochs: 1 full training iteration
    # batch size: # of samples through nn before we update parameters
    # lr: how much to update parameters at each batch/epoch
        # smaller lr is a slower learning speed, but bigger lrs may not produce the most accurate predictions

learning_rate = 1e-3 # this is a common lr
batch_size = 64
# epochs = 5 # NOTE: there is an error in this tutorial where this is defined here, but then later redefined when the actual training is done - commenting it out here so it can still be used as an example here

# Optimization Loop (1 loop = 1 epoch)
# each epoch consists of 2 main parts:
    # The training loop, where we try to converge to optimal parameters
    # The validation/test loop: iterate over the test dataset to check if model performance is improving

# Concepts in the training loop:

# Loss Function
# measures the degree of dissimilarity of obtained result to the target value - we want to MINIMIZE this during training
# to calculate the loss, we make a prediction using the inputs of our given data sample and compare it against the true data label
    # so you have your test images - you take one, make the guess; then you look at the correct label to see if you were right
    # this is why our test set is so much smaller than our training set, we just need a small amount of images to check our accuracy

# common loss functions include nn.MSELoss (Mean Square Error) for regression, and nn.LLLoss (Negative Log Likelihood) for classifcation
# here, we use nn.CrossEntropyLoss(), which combines nn.LogSoftmax and nn.NLLLoss
# here, we will set our CrossEntropyLoss() function equal to loss_fn for readability
    # this will make it so that in our implementation down the line, we don't need to call nn.CrossEntropyLoss(data), we just call loss_fn(data)

loss_fn = nn.CrossEntropyLoss()

# Optimizer
# we use optimization algorithms to optimize our parameters - here, we are using Stochastic Gradient Descent
# optimization logic is encapsulated in the optimizer object
# other optimizers (like ADAM and RMSProp) in pytorch are detailed here: https://pytorch.org/docs/stable/optim.html

# we initialize the optimizer by registering the model's parameters that need to be trained
# then, we pass in the learning rate
# recall from tutorial #4: our model keeps track of our parameters for us when we use nn.Module subclasses, so we just have to call model.parameters() here
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# inside the training loop, optimization happens in 3 steps:
    # call optimizer.sero_grad() to reset the gradients of model params to prevent sobule-counting (ie bc gradients add up, we "reset" them at each iteration)
    # backpropagate the predicition loss with loss.backward()
    # once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass

# here is the full implementation:

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Now that everything is defined, here is where we do our actual training!
# recall that we already defined our loss_fn and learning rate above; for clarity, you would probably define them just before calling the functions
# since we already defined them earlier in the tutorial, I'll leave them commented our here so that I remember to do this in the future as best-practice

# the following code defines our loss, optimizer based on params/lr, and how many epochs we want
# then, for each epoch, it will call the training loop and the testing loop
# when the model has finished training, it will print "Done!"

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


