import torch
								#TENSORS
### SCALAR
scalar = torch.tensor(7)
print(scalar.item())

### VECTOR
vector = torch.tensor([7,7])
print(vector.ndim)
print(vector.shape)

### MATRIX
MATRIX = torch.tensor([[1,2],
					   [3,4]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

### TENSOR
TENSOR = torch.tensor([[[1,2,3],
						[4,5,6],
						[7,8,9]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

### RANDOM TENSORS
random_tensor = torch.rand(1,3,3) # 2 dimentional
print(random_tensor)
print(random_tensor.ndim)

random_tensor_2 = torch.rand(1,3,3) # 3 dimentional
print(random_tensor_2)
print(random_tensor_2.ndim)

#Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) #height , width , colour(R,G,B)
print(random_image_size_tensor)
print(random_image_size_tensor.ndim)
print(random_image_size_tensor.shape)

### ZEROS AND ONES
#Crate a tensor of all zeros
zeros = torch.zeros(3,3) # or zeros = torch.zeros(size=(3,3))
print(zeros)

#Create a tensor of all ones
ones = torch.ones(3,3) # or ones = torch.ones(size=(3,3))
print(ones)
print(ones.dtype) # tensor data type

### CREATING A RANGE OF TENSORS AND TENSORS-LIKE
#Use torch.range
#// one_to_ten = torch.range(0,10) #!will be removed in a future , use "arange"!
one_to_ten = torch.arange(0,11)
print(one_to_ten)
start_step_end = torch.arange(start=0,end=1000,step=100)
print(start_step_end)

# Creating tensors like
ten_zeros = torch.zeros_like(one_to_ten) # or ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

### TENSOR DATATYPES
# **Note:** Tensor datatypes is one of the 3 big errors you'll run into with PyTorch & deep learning :
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tesors not on the right device

float_32_tensor = torch.tensor([1.0,2.0,3.0],dtype=None, # What datatype is the tensor (float32,fload16 ...)
											 device=None, # What device is your tensor on
											 requires_grad=False) # Whether or not to track gradients with this tensor operations
print(float_32_tensor)
print(float_32_tensor.dtype)

#float_16_tensor = torch.tensor([1.0,2.0,3.0],dtype=torch.float16)
# or
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)
print(float_16_tensor.dtype)

print(float_16_tensor * float_32_tensor)

### MANIPULATING TENSORS (tensor operations)
# Tensor operations include:
# * Addition
# * Subtraction
# * Multiplication 
# * Division
# * Matrix multiplication

tensor = torch.tensor([1,2,3])
tensor + 10
tensor * 10
tensor - 10
tensor / 10

# Matrix multiplication
# Two main ways of performing multiplication in neural networks and deep learning:
# 1. Element-wise multiplication
# 2. Matrix multiplication (dot product) -> https://www.mathsisfun.com/algebra/matrix-multiplying.html

# Element-wise multiplication
tensor_multipl = torch.tensor([1,2,3])
print(tensor_multipl*tensor_multipl)

# Matrix multiplication
tensor_matr_multipl = torch.matmul(tensor , tensor)
print(tensor_matr_multipl)
# in code it looks like so:
# //value=0
# //for i in range(len(tensor_multipl)):
# //	value += tensor_multipl[i] * tensor_multipl[i]
# //print(value)

tensor_multipl @ tensor_multipl # symbol "@" stands for matrix multiplication

# There are two main rules that performing matrix multiplication needs to satisfy:
# 1. The **inner dimensions** must match:
# * '(3 , 2) @ (3 , 2)' won't work
# * '(2 , 3) @ (3 , 2)' will work
# * '(3 , 2) @ (2 , 3)' will work
# 2. The resulting matrix has the shape of the **outer dimensions**:
# * '(2 , 3) @ (3 , 2)' -> '(2 , 2)'
# * '(3 , 2) @ (2 , 3)' -> '(3 , 3)'

matr_mul = torch.matmul(torch.rand(10,10) , torch.rand(10,10))
print(matr_mul)

# Shapes for matrix multiplication
tensor_A = torch.tensor([[1,2],
						 [3,4],
						 [5,6]])
tensor_B = torch.tensor([[7,10],
						 [8,11],
						 [9,12]])
# multiplied = torch.mm(tensor_A,tensor_B) # torch.mm is the same as torch.matmul

# to fix our tensor shape issues, we can manipulate the shape of one of our tensors using a **transpose**.
# A **transpose** switches the axes or dimensions of a given tensor. "tensor.T" is  transpose

#!!! The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release . So we'll use "x.mT"!!! (Warning is not always)

print(tensor_B)
print(tensor_B.shape)
transposed = tensor_B.T 
print(transposed)
print(transposed.shape)
multiplied = torch.matmul(tensor_A,transposed) # the matrix multiplication operation works when tensor_B is transposed
print(multiplied)

###Finding the min , max, mean(avarage), sum, etc(tensor aggregation)

x = torch.arange(10,101)
print(x)
#Find the min
print(torch.min(x),x.min())
#Find the max
print(torch.max(x),x.max())
#Find the mean, works either only with float and complex datatypes
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())
#Find a sum
print(torch.sum(x),x.sum()) 

### Finding the positional min and max
# Find the position in tensor that has the minimum value argmin() -> returns index position of target tensor where the minimu value occurs
print(torch.argmin(x), x.argmin())
# Find the position in tensor that has the minimum value argmax() -> returns index position of target tensor where the maximum value occurs
print(torch.argmax(x), x.argmax())


### RESHAPING , STACKING, SQUEEZING AND UNSQUEEZING TENSORS
# * Reshaping - reshapes an input tensor to a defined shape *
# * View - return a view of an input tensor of certain shape but keep the same memory as the original tensor *
# * Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack) *
# * Squeeze - removes all '1' dimensions from a tensors *
# * Unsqueeze - add a '1' dimension to a target tensor *
# * Permute - return a view of the input with dimensions permuted (swapped) in a certain way *

#let's crate a tensor
t = torch.arange(1.,11.)
print(t,t.shape)

#Add an extra dimesion
t_reshaped = t.reshape(2,5)
print(t_reshaped)
print(t_reshaped.shape)

#Change view
viewed_t = t.view(2,5)
print(viewed_t)
print(viewed_t.shape) # !!! under this is really important !!!

# Changing viewed_t changes t (because a view of a tensor shares the same memory as the original input)
# // viewed_t[:, 0] = 5 #changing the values -> https://www.geeksforgeeks.org/how-to-access-and-modify-the-values-of-a-tensor-in-pytorch/
# // print(viewed_t,t)

#Stack tensors on top of each other
t_stacked = torch.stack([t,t,t,t,t,t,t], dim=0)
print(t_stacked)

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous Tensor : {t_reshaped}")
print(f"Previous shape : {t_reshaped.shape}")
# remove an extra dimension
print(f"New Tensor : {t_reshaped.squeeze()}")
print(f"New Tensor : {torch.squeeze(t_reshaped)}")
print(f"New shape : {t_reshaped.squeeze().shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(f"Previous Target : {t_reshaped}")
print(f"Previous shape : {t_reshaped.shape}")
# add an extra dimension
print(f"New Target : {t_reshaped.unsqueeze(dim=0)}")
print(f"New Target : {torch.unsqueeze(t_reshaped,dim=0)}")
print(f"New shape : {t_reshaped.unsqueeze(dim=0).shape}")

# torch.permute - rearranges the dimensions of a target tensor in a specified order
t_original = torch.randn(3,6,9)
print(t_original.shape)
# permute the original tensor to rearange the axis (or dim) order
t_permuted = t_original.permute(2,0,1) 
print(t_permuted.shape)

### Indexing (selecting data from tensors)

# * Note * : indexing in PyTorch is similar to indexing in Numpy

# Create a tensor
i = torch.arange(1,10).reshape(1,3,3)
print(i, i.shape)
# Let's ondex on a new tensor
print(i[0])
# Let's ondex on a middle bracket(dim=0)
print(i[0][0])

print(i[0][0][0])

# in details -> https://www.geeksforgeeks.org/how-to-access-and-modify-the-values-of-a-tensor-in-pytorch/
# " : " Access all values of only certain column
print(i[:, 1])
#Get all values of 0th and 1st dimentions but only index 1 of 2nd dimension
print(i[:,:,1])
#Get all values of 0th dimention but only the 1 index value of 1st and 2nd dimenison
print(i[:,1,1])
#Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(i[0,0,:])

# Index on x to return 9
print("Index on x to return 9")
print(i[0][2][2])

# Index on x to return 3,6,9
print("Index on x to return 3,6,9")
print(i[:,:,2])

### 						PyTorch Tensors and NumPy
# Numpy is a popular scientific Python numerical computing library
# And because of this , PyTorch has functionality to interact with it
# * Data in Numpy, want in PyTorch tensor -> <torch.from_numpy(ndarray)> 
# * PyTorch tensor -> Numpy -> <torch.Tensor.numpy()>

# NumPy array to tensor
import numpy as np
numpy_array = np.arange(1.0,11.0)
tensor_array = torch.from_numpy(numpy_array) # ! when converting from numpy dtype=float64 !
print(numpy_array,tensor_array)

# Change the value of array , what will this do to 'tensor_array'
numpy_array += 1
print(numpy_array)
print(tensor_array)

# Tensor to NumPy array
to_np = torch.ones(3,3).type(torch.float32) # we changed data type of tensor to check if it will be the same in numpy array
np_array = torch.Tensor.numpy(to_np)
print(np_array.dtype)
print(to_np.numpy())
print(to_np)
print(np_array)

# Change the tensor , what will this do to 'np_array'
to_np[:,2] = 0
print(to_np,np_array)


### Reproducibility (trying to take random out of random)
# In short how neural network learns:
# strat with random numbers -> tensor operations -> update random numbers to try and make them
# better representatinos of the data -> again -> again -> ...

# To reduce the randomness in neural networks and PyTorch comes the concept of ** random seed **.
# Essentially what a random seed does is "flavour" the randomness .

# Create two random tensors
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random bur reproducible tensors
#Sat the random seed
RANDOM_SEED = 100

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
# Extra resources for reproducibility:
# * https://pytorch.org/docs/stable/notes/randomness.html
# * https://en.wikipedia.org/wiki/Random_seed