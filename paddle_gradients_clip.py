import paddle

# clip gradient by value

# clip all gradient
linear = paddle.nn.Linear(10, 10)
clip = paddle.nn.ClipGradByValue(min=-1, max=1)
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

# clip a part of gradients
linear = paddle.nn.Linear(10, 10，bias_attr=paddle.ParamAttr(need_clip=False))

# clip gradient by norm

# clip al gradient
linear = paddle.nn.Linear(10, 10)
clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

# clip a part of gradients
linear = paddle.nn.Linear(10, 10, weight_attr=paddle.ParamAttr(need_clip=False))


# clip gradient by global norm
linear = paddle.nn.Linear(10, 10)
clip = paddle.nn.ClipGradByGloabalNorm(clip_norm=1.0)
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)


