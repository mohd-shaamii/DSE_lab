import random
import matplotlib.pyplot as plt

def lineargradient(x, y, theta):
    m, b = theta
    predicted = m * x + b
    error = predicted - y
    grad = [2 * error * x, 2 * error]
    return grad

def meangradient(data, theta):
    gradients = [lineargradient(x, y, theta) for x, y in data]
    return [sum(gradient[i] for gradient in gradients) / len(gradients) for i in range(len(theta))]

def gradientdescent(data, theta, learningrate, epochs):
    plt.figure(figsize=(10, 5))

    for epoch in range(epochs):
        grad = meangradient(data, theta)
        theta = [theta[i] - learningrate * grad[i] for i in range(len(theta))]

        xvalues = [x for x, _ in data]
        yvalues = [y for _, y in data]
        plt.scatter(xvalues, yvalues, label='original data')
        linex = [min(xvalues), max(xvalues)]
        liney = [theta[0] * x + theta[1] for x in linex]

        plt.plot(linex, liney, color='red', label='linear regression line')
        plt.quiver(theta[0], theta[1], -grad[0], -grad[1], angles='xy', scale_units='xy', scale=1, color='green', width=0.01)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Epoch {epoch + 1}')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()

    return theta

def main():
    data = [(1, 3), (2, 5), (3, 7), (4, 9)]
    initial_theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.1
    num_epochs = 20
    final_theta = gradientdescent(data, initial_theta, learning_rate, num_epochs)
    print("final parameters", final_theta)

if __name__ == "__main__":
    main()
