# Foundation

A tiny neural net library for fun and educational purposes inspired by the work of A. Karpathy.

## Example Usage

The following code implements a multi-layer perceptron as depicted below:

![foundation](assets/neural_net.jpg)
Source: https://cs231n.github.io/convolutional-networks

```python
# multi-layer perceptron with 3 inputs, 3 layers with 4, 4 and 1 neurons
model = MLP(no_inputs=3, no_layer_outputs=[4, 4, 1])
model.summary()
```

Output:

```bash
===== Model Summary =====
1. Dense Layer of 4 Tanh-Neurons: 16 params
2. Dense Layer of 4 Tanh-Neurons: 20 params
3. Dense Layer of 1 Tanh-Neurons: 5 params
=========================
Total trainable parameters: 41
```

### Training

```python
# input
xs = [
    [1.0, 4.0, -1.0],
    [2.0, -2.0, 0.5],
    [0.5, 1.0, 3.0],
    [3.0, 1.0, -1.0]
]

# labels aka desired targets
ys = [1.0, -1.0, -1.0, 1.0]

optimizer = SGD(learning_rate=0.05)
history = model.fit(x=xs, y=ys, optimizer=optimizer, epochs=200)
```

History:
```bash
epoch 0 loss: 5.765349300153446
epoch 1 loss: 3.250696615211469
epoch 2 loss: 2.7737727679400273
epoch 3 loss: 2.2183151169026636
...
epoch 196 loss: 0.0029259165654474954
epoch 197 loss: 0.002909801686845781
epoch 198 loss: 0.002893856798439156
epoch 199 loss: 0.00287807925246501
```

### Inference
```
predictions = [model(x) for x in xs]
```

Predictions:

```
[
    [Scalar(data=0.9801648697903278)],
    [Scalar(data=-0.9642723731566557)],
    [Scalar(data=-0.9830055518414185)],
    [Scalar(data=0.9699374073460569)]
]
```

Model Graph:
```python
draw_graph(predictions)
```
![foundation](assets/graph.svg)
