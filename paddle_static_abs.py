import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import numpy as np


paddle.enable_static()

shape = [9, 10]
place = fluid.CPUPlace()
dtype = 'float32'

x_data = np.random.random(size=shape).astype(dtype)
x_var = paddle.static.create_global_var(
    name='x', shape=shape, value=0.0, dtype=dtype, persistable=True)
out = paddle.abs(x_var)


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

fluid_out = exe.run(
    fluid.default_main_program(),
    feed={'x': x_data},fetch_list=[out])



