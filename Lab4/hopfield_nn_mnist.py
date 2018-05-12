from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Hopfield:

    def __init__(self):
        self.data = None
        self.bias = 0
        self.dim = 0
        self.n_patterns = 0
        self.weights = None

    def initialize_weights(self):
        self.weights = np.zeros((self.dim, self.dim))
        self.n_patterns = len(self.data)
        mean = np.sum([np.sum(x) for x in self.data])/(self.n_patterns * self.dim)

        # Compute weight matrix as outer product of normalized sample data
        for i in range(self.n_patterns):
            t = self.data[i] - mean
            self.weights += np.outer(t, t)

        # Values on a diagonal need to be zeros
        for i in range(self.dim):
            self.weights[i, i] = 0

        self.weights /= self.n_patterns

    def learn(self, patterns, bias):
        self.data = patterns
        self.bias = bias
        self.dim = len(patterns[0])
        self.initialize_weights()

    def predict(self, signals):
        outputs = np.zeros(np.shape(signals))

        for i, signal in enumerate(signals):
            outputs[i] = self.weights.dot(signal) - self.bias

        return np.sign(outputs)


class Mnist:

    def __init__(self, digit=0, n_patterns=10):
        self.n = int(digit*7000)
        self.data = self.get_data(n_patterns)

    def get_data(self, patterns):
        mnist = fetch_mldata('MNIST original', data_home='mnist/')
        mnist.data = mnist.data.astype(np.float32)
        mnist.data /= 255

        return mnist.data[self.n:self.n+patterns]

    def add_noise(self, error_rate):
        data = self.data[:]
        for i, t in enumerate(data):
            s = np.random.binomial(1, error_rate, len(t))
            for j in range(len(t)):
                if not s[j]:
                    t[j] *= -1
        return data


def plot_pictures(pic1, pic2, label1='', label2=''):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(pic1, cmap='gray')
    ax.set_title(label1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(pic2, cmap='gray')
    ax2.set_title(label2)
    plt.show()


if __name__ == '__main__':

    samples = 115
    test_cases = 10
    model = Hopfield()
    mnist = Mnist(digit=5, n_patterns=samples)

    train_data = mnist.data[:]
    test_data = mnist.add_noise(error_rate=0.2)[:test_cases]

    model.learn(train_data, bias=0)
    predictions = model.predict(test_data)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    def animate(i):
        ax1.clear()
        ax2.clear()

        ax1.imshow(np.reshape(test_data[i], (28, 28)), cmap='gray')
        ax1.set_title('Signal')
        ax2.imshow(np.reshape(predictions[i], (28, 28)), cmap='gray')
        ax2.set_title('Prediction')


    ani = animation.FuncAnimation(fig, animate, np.arange(0, test_cases, 1), interval=1, repeat_delay=100)
    plt.show()




