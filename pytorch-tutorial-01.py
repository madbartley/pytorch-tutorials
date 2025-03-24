

import torch
import numpy as np

### Initializing a tensor
# "tensors are multi-dimensional arrays that contain floating-point, intger, or boolean data" -taken from this video: https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
# Tensors can be created directly from data, where the data type is automatically inferred
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print("Printing my first tensor:", end = '\n')
print(x_data, end = '\n\n')

print("Printing the lists in my tensor:", end = '\n')
for i in range(len(x_data)):
    print(i, ": ", x_data[i])

print('\n')

print("Printing each individual item:", end = '\n')
for i in range (len(x_data)):
    for j in range (len(x_data)):
        print(x_data[i][j])

# tensor from a NumPy array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print('\n')

print("Printing a numpy array converted from my tensor:")
print(np_array, end = '\n')

print('\n')

print("Printing my torch tensor converted from a numpy array: ")
print(x_np)

print('\n')

# Creating a tensor from another tensor
# Note: the new tensor retains the properties, like shape and datatype, of the argument tensor unless overridden

# recall, x_data is our original tensor
# torch.ones_like() generates a tensor with identical shape and data types as the x_data, but all values = 1
x_ones = torch.ones_like(x_data)

print("Printing a tensor of shape (2,2) and datatype integers:")
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype = torch.float)

print("Printing a tensor of shape (2,2) but with float datatype")
print(f"Random Tensor: \n {x_rand} \n")

# Note: we can get the shape of a tensor by the following:
print("Printing the shape of my x_data tensor: ")
print(x_data.size())

print('\n')

### Getting the shape of a tensor
# Shape of a tensor: (rows, columns)
# the size() method, demonstrated above, returns a torch.Size object, which is a subclass of a tuple (in Python, an ordered, immutable sequence of elements of different datatypes)
# unlike a tuple, the class has additional methods:

my_tensor = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])

print("Printing the size of a (3,3) tensor")
print(my_tensor.size())

print('\n')

# to convert our size object to a list of integers, use list()

shape = my_tensor.size()
shape_list = list(shape)

print("Printing the shape of my object as a list: ")
print(shape_list)

print('\n')

# .size() vs .shape
#  Note: no parentheses on .shape! And, both of these options will be fed into list() to print a list

shape_alt = my_tensor.shape
shape_list_alt = list(shape_alt)

print("Printing the shape of my tensor using .shape instead of .size(): ")
print(shape_list_alt)

# Getting the shape with random or constant values

# first, establish the shape that you want
shape = (2,3)

# here, we create a 2,3 tensor with random values, with ones as values, and with zeros as values
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# this will print our tensors from above
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

print('\n')

### Attributes of a tensor
# Tensor attributes describe shape, datatype, and the device on which they are stored (like the CPU, cuda, or other; you can read more about devices here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)

# first create a tensor with random values, of 3 rows, 4 columns
tensor = torch.rand(3,4)

# now we can print the attributes of the tensor:
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print('\n')


### Operations on tensors, and using Accelerators
# 1200 tensor operations described in detail here: https://pytorch.org/docs/stable/torch.html
# Each of these operations can be run on the CPU and Accelerator such as CUDA, MPS, MTIA, or XPU. If you’re using Colab, allocate an accelerator by going to Runtime > Change runtime type > GPU.
# read more about Accelerators here: https://pytorch.org/docs/stable/torch.html#accelerators
# DEFAULT device for tensors: CPU - to move tensors to an accelerator, use the .to method, after checking for accelerator availability

# check for accelerator availability, then move to accelerator if available
# I do not have an accelerator available, so I'm commenting this function out

# if torch.accelerator.is_available():
    #tensor = tensor.to(torch.accelerator.current_accelerator())


# trying out some operations on our tensor
# recall: tensor = torch.rand(3,4)
print("A randomized tensor: \n" )
print(tensor) #note: this will change every time you run the program, so do not depend on these numbers

print('\n')

# is_tensor: returns True if object is a PyTorch tensor
print(f"testing is_tensor: \n{torch.is_tensor(tensor)}\n")

# argwhere: returns a tensor containing the indices of all non-zero elements of the input
argwhere_tensor = torch.argwhere(tensor)
print(f"testing argwhere: \n{argwhere_tensor}\n")

# column_stack: creates a new tensor by horizontally stacking the tensors in the input tensors
# note: the input tensors MUST be in the form of a tuple of 2+ tensors, hence the creation of tensor2 and tensor_tuple below
tensor2 = torch.rand(3,4)
print("Printing our two randomized tensors for the following operation tests")
print(tensor, '\n')
print(tensor2, '\n')

tensor_tuple = (tensor, tensor2)
column_stack_tensor = torch.column_stack(tensor_tuple)
print(f"testing column_stack: \n{column_stack_tensor}\n")

# hstack: similar to column_stack, stacks tensors in sequence horizontally, column-wise
h_stack_tensor = (tensor, tensor2)
print(f"tsting hstack: \n{h_stack_tensor}\n")

# gather: gathers values along an axis specified by "dim", and according to an index tensor of the same dimension as the input tensor
# this one is a bit more involved than the previous operations
# note: words like "rank" and "axis" are different in regards to TENSORS than they are in traditional mathematics (ie when they describe matrices, vectors, etc)
    # Rank: number of DIMENSIONS (aka AXES) present in a tensor
        # a rank = 2 tensor means that we have a matrix, a 2D array, and a 2D tensor (think: an array of arrays)
        # *** important: the RANK tells us how many INDEXES are needed to access a single element ie element[i][j] will give you a single char in a char array of arrays, while element[i] will give you a full array
    # Axis: in regards to tensors, an axis refers to the specific dimension of a tensor
        # this is different than axes x, y, z in 3D space
        # the length of the axis is the number of elements contained in each axis
            # so for example, in a rank = 2 tensor, let's say we have the SHAPE 3, 4
            # this means that we have 3 arrays and each array has 4 elements - we can also call these rows and columns
            # so an example of that tensor might look like this:
                # [ [1, 2, 3, 4]
                #   [4, 5, 6, 7]
                #   [8, 9, 10, 11] ]
            # this tensor can be thought of as an array of arrays - the outer square brackets denote the first array, and that array contains 3 more arrays, each with 4 elements
            # the first axis, therefore is 3 (where each element is an array), and the second axis is 4 (where each element is a single integer)
            # as you might have noticed, the SHAPE of a tensor, discussed earlier in this tutorial, can be said to be made up of the length of each axis
            
# going back to our gather operation, we are using our randomized tensor called "tensor", which is of dimension 2, shape (3, 4)
# now we need to decide which dimension or axis we want to look at - our tensor is of dimension 2, so we can use either 1 or 2
# we also need a new tensor to act as our index; this index tensor will tell us which elements to select, so we need to make sure that tensor and index_tensor are of the same dimension
    # recall: this is because our index tensor will be used to "index into" our tensor, so for instance, we could not index into the third dimension of a 2D array
tensor_index = torch.tensor([[0,1,2]])
print(tensor)
print(f"testing gather: {torch.gather(input=tensor, dim=0, index=tensor_index)}\n")

# this is equivalent to us saying "for the 0th dimension (our arrays), give me the 0th, the 1st, and the 2nd item in the 0th, the 1st, and the 2nd array"
# in this way, we are "gathering" items along a dimension

# Standar numpy-like indexing and slicing

# creating a new randomized tensor for practice
tensor = torch.rand(4,4)

print(f"Printing our random tensor before slicing/indexing: \n{tensor}\n")

print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0

print(f"Printing our new tensor: \n{tensor}\n")


# joining tensors using torch.cat (as in, contatenate)
# you can use torch.cat to join a sequence of tensors along a given dimension (or axis)

# here, t1 will be a tensor of our randomized tensor repeating itself 3 times
t1 = torch.cat([tensor, tensor, tensor])
print(f"Printing our joined randomized tensor: \n{t1}\n")

# arithmetic operations on tensors

# matrix multiplication between two tensors
    # the following examples are different ways to accomplish matrix multiplication
# "tensor.T" will return the transposed tensor
m1 = tensor @ tensor.T

m2 = tensor.matmul(tensor.T)

m3 = torch.rand_like(m1)
torch.matmul(tensor, tensor.T, out = m3) # note the similarities between this syntax and the syntax for m2

print(f"Printing our multiplied matrices, which should all be the same: \n{m1}\n\n{m2}\n\n{m3}\n\n")

# compute the element-wise product, where z1, z2, z3 will all have the same value
z1 = tensor * tensor

z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3) # note the similarities between this syntax and the syntax for z2

print(f"Printing our multiplied tensors which should all be the same: \n{z1}\n\n{z2}\n\n{z3}\n\n")


# Single-element tensors
# if  you have a single-element tensor, you can convert it to a Python numerical value using item()
# here, we aggregate all values of a tensor into one single value

agg = tensor.sum()
agg_item = agg.item()
print("Our single value:", agg_item, type(agg_item), end = '\n\n')

# In-place operations
# operations that store the result into the operant are called in-place. They are denoted by an underscore suffix, like so: x.copy_(y) and x.t_()
# in the examples in the last line, "x" will be changed
# note: in-place operations save memory, but are problematic for certain procedures like calculating derivatives - their use is generally discouraged in best practices

print(f"Our tensor before an in-place operations: \n{tensor}\n")
tensor.add_(5)
print(f"Our tensor AFTER an in-place operation to add 5: \n{tensor}\n")


### Bridge to NumPy

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}\n")

n = t.numpy()
print(f"n: {n}\n")

# now that we have converted our tensor to numpy, if we change the tensor, the numpy array is already changed
t.add_(1)
print(f"our new t: {t}\n")
print(f"our similarly changed n: {n}\n\n")

# working the opposite direction - numpy to tensor:
print("Working backwards: \n")
n = np.ones(5)
print(f"n: {n}\n")

t = torch.from_numpy(n)
print(f"t: {t}\n")

# now we change the numpy array to see the reflection in the tensor
np.add(n, 1, out = n)
print(f"our torch tensor, changed by adding 1 to our numpy array: {n}\n")
print(f"the numpy array that we added 1 to: {n}\n\n")



