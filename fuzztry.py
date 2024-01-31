import random
import numpy as np
from typing import List
import tqdm
import math

def squared_distance(prediction, target):
    return np.sum((np.array(prediction) - np.array(target))**2)

def fizzbuzzencode(x) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def feed_forward(network, input_vector):
    outputs = []
    for layer in network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def gradient_step(v_list: List[int], gradient: List[int], step_size: int) -> List[int]:
    """Moves `step_size` in the `gradient` direction from `v_list`"""
    # assert len(v_list) == len(gradient)
    step =(int) [step_size * grad_i for grad_i in gradient]
    return [v_i + step_i for v_i, step_i in zip(v_list, step)]

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

def binaryencode(x: int) -> List[int]:
    binary = []
    for i in range(10):
        binary.append(x % 2)
        x = x // 2
    return binary

def sqerror_gradients(network, x, y):
    hidden_outputs = feed_forward(network[:-1], x)[0]
    output_predictions = feed_forward(network, x)[-1]
    output_gradients = [(2 * output_predictions[i] - y[i]) * output_predictions[i] * (1 - output_predictions[i]) for i in range(len(y))]
    hidden_gradients = [hidden_outputs[j] * (1 - hidden_outputs[j]) * np.dot([network[-1][i][j] * output_gradients[i] for i in range(len(output_gradients))], output_gradients) for j in range(len(hidden_outputs))]
    return [list(hidden_gradients), list(output_gradients)]

xs = [binaryencode(n) for n in range(101, 1024)]
ys = [fizzbuzzencode(n) for n in range(101, 1024)]
HIDDEN = 25
network = [
    [[random.random() for _ in range(11)] for _ in range(HIDDEN)],
    [[random.random() for _ in range(HIDDEN + 1)] for _ in range(4)]
]
learning_rate = 0.01

with tqdm.trange(10) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(np.array(predicted), np.array(y))
            gradients = sqerror_gradients(network, x, y)
            network = [
                [
                    gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)
                ]
                for layer, layer_grad in zip(network, gradients)
            ]
        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")

def argmax(xs: List) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])

num_correct = 0
for n in range(1, 101):
    x = binaryencode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizzbuzzencode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    if predicted == actual:
        num_correct += 1

print(num_correct, "/", 100)
