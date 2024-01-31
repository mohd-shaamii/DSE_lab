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
    return [hidden_gradients, output_gradients]

def gradient_step(neuron, grad, learning_rate):
    if isinstance(grad, np.ndarray):
        grad = grad.item()
    return [weight - learning_rate * grad for weight in neuron]

xs = [binaryencode(n) for n in range(101, 1024)]
ys = [fizzbuzzencode(n) for n in range(101, 1024)]
HIDDEN = 25
network = [
    [[random.random() for _ in range(11)] for _ in range(HIDDEN)],
    [[random.random() for _ in range(HIDDEN + 1)] for _ in range(4)]
]
learning_rate = 0.01

with tqdm.trange(500) as t:
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



# OUTPUT:
# /lib/python3.11/site-packages/tqdm/std.py:1525: TqdmMonitorWarning: tqdm:disabling monitor support (monitor_interval = 0) due to:
# can't start new thread
#   return tqdm(range(*args), **kwargs)
# fizz buzz (loss: 2768.98): 100%|██████████| 500/500 [05:40<00:00,  1.47it/s]

# 1 fizz 1
# 2 fizz 2
# 3 fizz fizz
# 4 fizz 4
# 5 fizz buzz
# 6 fizz fizz
# 7 fizz 7
# 8 fizz 8
# 9 fizz fizz
# 10 fizz buzz
# 11 fizz 11
# 12 fizz fizz
# 13 fizz 13
# 14 fizz 14
# 15 fizz fizzbuzz
# 16 fizz 16
# 17 fizz 17
# 18 fizz fizz
# 19 fizz 19
# 20 fizz buzz
# 21 fizz fizz
# 22 fizz 22
# 23 fizz 23
# 24 fizz fizz
# 25 fizz buzz
# 26 fizz 26
# 27 fizz fizz
# 28 fizz 28
# 29 fizz 29
# 30 fizz fizzbuzz
# 31 fizz 31
# 32 fizz 32
# 33 fizz fizz
# 34 fizz 34
# 35 fizz buzz
# 36 fizz fizz
# 37 fizz 37
# 38 fizz 38
# 39 fizz fizz
# 40 fizz buzz
# 41 fizz 41
# 42 fizz fizz
# 43 fizz 43
# 44 fizz 44
# 45 fizz fizzbuzz
# 46 fizz 46
# 47 fizz 47
# 48 fizz fizz
# 49 fizz 49
# 50 fizz buzz
# 51 fizz fizz
# 52 fizz 52
# 53 fizz 53
# 54 fizz fizz
# 55 fizz buzz
# 56 fizz 56
# 57 fizz fizz
# 58 fizz 58
# 59 fizz 59
# 60 fizz fizzbuzz
# 61 fizz 61
# 62 fizz 62
# 63 fizz fizz
# 64 64 64
# 65 65 buzz
# 66 66 fizz
# 67 fizz 67
# 68 68 68
# 69 fizz fizz
# 70 fizz buzz
# 71 fizz 71
# 72 fizz fizz
# 73 fizz 73
# 74 fizz 74
# 75 fizz fizzbuzz
# 76 fizz 76
# 77 fizz 77
# 78 fizz fizz
# 79 fizz 79
# 80 fizz buzz
# 81 fizz fizz
# 82 fizz 82
# 83 fizz 83
# 84 fizz fizz
# 85 fizz buzz
# 86 fizz 86
# 87 fizz fizz
# 88 fizz 88
# 89 fizz 89
# 90 fizz fizzbuzz
# 91 fizz 91
# 92 fizz 92
# 93 fizz fizz
# 94 fizz 94
# 95 fizz buzz
# 96 96 fizz
# 97 fizz 97
# 98 fizz 98
# 99 fizz fizz
# 100 fizz buzz
# 27 / 100

