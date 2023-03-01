import paddle
import numpy as np
"""
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y

model = Model()
print(model.sublayers())

print("--------------------------")

for item in model.named_sublayers():
    print(item)

fc = paddle.nn.Linear(10, 3)
model.add_sublayer("fc", fc)
print(model.sublayers())


def function(layer):
    print(layer)

model.apply(function)


sublayer_iter = model.children()
for sublayer in sublayer_iter:
    print(sublayer)
"""


"""
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        img = self.create_parameter([1,3,256,256])
        self.add_parameter("img", img)
        self.flatten = paddle.nn.Flatten()

    def forward(self):
        y = self.flatten(self.img)
        return y

model = Model()
model.parameters()

print('-----------------------------------------------------------------------------------')

for item in model.named_parameters():
    print(item)


model = Model()
out = model()
out.backward()
model.clear_gradients()

"""

"""
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.saved_tensor = self.create_tensor(name="saved_tensor0")
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)

    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y
    
"""


"""

class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        saved_tensor = self.create_tensor(name="saved_tensor0")
        self.register_buffer("saved_tensor", saved_tensor, persistable=True)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)

    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y

model = Model()
print(model.buffers())

print('-------------------------------------------')

for item in model.named_buffers():
    print(item)
    
"""


# execute a layer

class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y

# configure execution mode
x = paddle.randn([10, 1], 'float32')
model = Model()
model.eval()  # set model to eval mode
out = model(x)
model.train()  # set model to train mode
out = model(x)
print(out)


# perform an execution
model = Model()
x = paddle.randn([10, 1], 'float32')
out = model(x)
print(out)

# add extra execution logic
# post hook
def forward_post_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_post_hook_handle = model.flatten.register_forward_post_hook(forward_post_hook)
out = model(x)
print(out)

# pre hook

def forward_pre_hook(layer, input):
    print(input)
    return input

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
out = model(x)

# save model's data
model = Model()
state_dict = model.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")

#load model's data back
model = Model()
state_dict = paddle.load("paddle_dy.pdparams")
model.set_state_dict(state_dict)









