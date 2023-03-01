# create **1-D Tensor** like matrix
import paddle
# The following sample code has imported the paddle module by default

# Creation of Tensor
# create by specifying data list
ndim_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0])
print(ndim_1_tensor)

print(paddle.to_tensor(2))
print(paddle.to_tensor([2]))

# create **2-D Tensor** like matrix
ndim_2_tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_tensor)


# create multidimensional Tensor
ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_tensor)


# create by specifying shape
#paddle.zeros([m, n])             # create Tensor of all elements: 0, Shape: [m, n]
#paddle.ones([m, n])              # create Tensor of all elements: 1, Shape: [m, n]
#paddle.full([m, n], 10)          # create Tensor of all elements: 10, Shape: [m, n]


# create by specifying interval
#t1 = paddle.arange(start, end, step)  # create Tensor within interval [start, end) evenly separated by step
#t2 = paddle.linspace(start, stop, num) # create Tensor within interval [start, stop) evenly separated by elements number


# Attributes of Tensor

# shape of tensor

ndim_4_tensor = paddle.ones([2, 3, 4, 5])
print("Data Type of every element:", ndim_4_tensor.dtype)
print("Number of dimensions:", ndim_4_tensor.ndim)
print("Shape of tensor:", ndim_4_tensor.shape)
print("Elements number along axis 0 of tensor:", ndim_4_tensor.shape[0])
print("Elements number along the last axis of tensor:", ndim_4_tensor.shape[-1])


ndim_3_Tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]],
                                  [[21, 22, 23, 24, 25],
                                   [26, 27, 28, 29, 30]]])
print("the shape of ndim_3_Tensor:", ndim_3_Tensor.shape)

reshape_Tensor = paddle.reshape(ndim_3_Tensor, [2, 5, 3])
print("After reshape:", reshape_Tensor.shape)

print("Tensor flattened to Vector:", paddle.reshape(ndim_3_tensor, [-1]).numpy())

# dtype of Tensor
print("Tensor dtype from Python integers:", paddle.to_tensor(1).dtype)
print("Tensor dtype from Python floating point:", paddle.to_tensor(1.0).dtype)

ndim_2_tensor = paddle.to_tensor([[(1+1j), (2+2j)],
                                  [(3+3j), (4+4j)]])
print(ndim_2_tensor)


float32_tensor = paddle.to_tensor(1.0)

float64_tensor = paddle.cast(float32_tensor, dtype='float64')
print("Tensor after cast to float64:", float64_tensor.dtype)

int64_tensor = paddle.cast(float32_tensor, dtype='int64')
print("Tensor after cast to int64:", int64_tensor.dtype)



# place of tensor on cpu
cpu_Tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_Tensor.place)

# place of tensor on gpu
#gpu_Tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
#print(gpu_Tensor.place) # the output shows that the Tensor is On the 0th graphics card of the GPU device

# place of tensor on pinned memory
#pin_memory_Tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
#print(pin_memory_Tensor.place)

# name of tensor
print("Tensor name:", paddle.to_tensor(1).name)


# method of tensor

# index and slice
ndim_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", ndim_1_tensor.numpy())
print("First element:", ndim_1_tensor[0].numpy())
print("Last element:", ndim_1_tensor[-1].numpy())
print("All element:", ndim_1_tensor[:].numpy())
print("Before 3:", ndim_1_tensor[:3].numpy())
print("From 6 to the end:", ndim_1_tensor[6:].numpy())
print("From 3 to 6:", ndim_1_tensor[3:6].numpy())
print("Interval of 3:", ndim_1_tensor[::3].numpy())
print("Reverse:", ndim_1_tensor[::-1].numpy())
t = ndim_1_tensor[:3]
t = paddle.to_tensor([2, 2, 2])
print("Origin Tensor:", ndim_1_tensor.numpy())


ndim_2_tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", ndim_2_tensor.numpy())

print("First row:", ndim_2_tensor[0].numpy())
print("First row:", ndim_2_tensor[0, :].numpy())
print("First column:", ndim_2_tensor[:, 0].numpy())
print("Last column:", ndim_2_tensor[:, -1].numpy())
print("All element:", ndim_2_tensor[:].numpy())
print("First row and second column:", ndim_2_tensor[0, 1].numpy())

# modify tensor
import numpy as np

x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32)) # [[1., 1., 1.], [1., 1., 1.]]

x[0] = 0                      # x : [[0., 0., 0.], [1., 1., 1.]]
x[0:1] = 2.1                  # x : [[2.09999990, 2.09999990, 2.09999990], [1., 1., 1.]]
x[...] = 3                    # x : [[3., 3., 3.], [3., 3., 3.]]

x[0:1] = np.array([1,2,3])    # x : [[1., 2., 3.], [3., 3., 3.]]

x[1] = paddle.ones([3])       # x : [[1., 2., 3.], [1., 1., 1.]]

x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")

print(paddle.add(x, y), "\n") # 1st method
print(x.add(y), "\n") # 2nd method

# Convert tensor to numpy array
tensor_to_convert = paddle.to_tensor([1.,2.])
tensor_to_convert.numpy()


# Convert numpy array to tensor
tensor_temp = paddle.to_tensor(np.array([1.0, 2.0]))
print(tensor_temp)


# broadcast
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# Two tensor have some shapes are broadcastable
z = x + y
print(z.shape)
# [2, 3, 4]

x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))

# compare from backward to forward：
# 1st step：y's dimention is 1
# 2nd step：x's dimention is 1
# 3rd step：two dimentions are the same
# 4st step：y's dimention does not exist
# So, x and y are broadcastable
z = x + y
print(z.shape)
# [2, 3, 4, 5]

# In Compare
# x = paddle.ones((2, 3, 4))
# y = paddle.ones((2, 3, 6))
# x and y are not broadcastable because in first step form tail, x's dimention 4 is not equal to y's dimention 6
# z = x, y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.



x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 1))
z = x + y
print(z.shape)
# z'shape: [2, 3, 4]

x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 2))
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
