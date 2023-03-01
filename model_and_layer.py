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



print("--------------------------")

sublayer_iter = model.children()
for sublayer in sublayer_iter:
    print(sublayer)



