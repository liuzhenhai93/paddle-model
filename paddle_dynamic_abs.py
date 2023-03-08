import paddle
import numpy as np

a = paddle.to_tensor(np.array([1.0, -2.0, 3]))
print("before abs {}".format(a))
t = paddle.abs(a)
print("after abs {}".format(t))
