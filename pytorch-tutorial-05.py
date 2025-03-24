

# Automatic Differentiation with torch.autograd

# algorithm: back propagation
# parameters (model weights) are adjusted according to the gradient of the los function with respect to the given parameter

# torch.autograd is a differentiation engine and supports automatic computation of gradient for any computational graph

# consider the simplest 1-layer nn, with input x, paramters w and b, and some loss function
# we define it as follows:
import torch

x = torch.ones(5)  # input tensor, a tensor of all 1's
y = torch.zeros(3)  # expected output, a tensor of all 0's
w = torch.randn(5, 3, requires_grad=True) # parameter
b = torch.randn(3, requires_grad=True) # parameter
z = torch.matmul(x, w)+b # torch.matmul() is matrix multiplication
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # defining our loss variable
# the tutorial has a cool image showing the computational graph here: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

# in the previous lines, we are trying to OPTIMIZE the parameters w and b
# to do this, we compute the gradients of loss function with respect to those variables, which is why we set the requires_grad property to True
# You can set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True) method.

# the function we use to construct the computational graph works forwards, but also computes its derivative during the backward propagation step
#  A reference to the backward propagation function is stored in grad_fn property of a tensor
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients
# again, we need to compute the derivatives of our loss functions w/r to w, b, all under some fixed values of x and y
# to do this, we use loss.backwards() and then retrieve values from w.grad and b.grad
loss.backward()
print(w.grad)
print(b.grad)

# NOTE: We can only obtain the grad properties for the leaf nodes of the computational graph, which have requires_grad property set to True. For all other nodes in our graph, gradients will not be available.
# NOTE: We can only perform gradient calculations using backward once on a given graph, for performance reasons. If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.


# Disabling Gradient Tracking
# for tensors that have requires_grad=True, their computational history is tracked
# however, that is not always necessary, for example when we jave a save model and just to apply it to some input data (ie we only want to do forward computations through the network)
# some other cases for disabling gradient tracking is to apply some parameters as "frozen parameters" or to speed up computations when you are only doing forward passing
# to do this, we surround the computation code with torch.no_grad()

z = torch.matmul(x, w)+b
print("Does z require gradient?: ", z.requires_grad)

# recall: "with" keyword will wrap execution of a block of code with a method, in this case, torch.no_grad()
with torch.no_grad():
    z = torch.matmul(x, w)+b
print("With our no_grad, does z require gradient?: ", z.requires_grad)

# you can also use the "detach()" method on the tensor:
z = torch.matmul(x, w)+b
z_det = z.detach()
print("Does z require gradient?: ", z_det.requires_grad)




