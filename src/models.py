import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, path, learning_rate=0.1, n_iterations=500, scaling_method='normalize'):
        self.theta = np.zeros((2, 1))
        self.x_min = None
        self.x_max = None
        self.x_mean = None
        self.x_std = None
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.cost_history = np.zeros(n_iterations)
        self.scaling_method = scaling_method

        self.data = np.loadtxt(open(path), delimiter=",", skiprows=1)
        x = self.data[:, 0]
        y = self.data[:, 1]
        self.original_x = x.reshape(x.shape[0], 1)
        self.y = y.reshape(y.shape[0], 1)

        self.scaled_data = self.scale_dataset()
        x = self.scaled_data[:, 0]
        self.scaled_x = x.reshape(x.shape[0], 1)
        self.x_oned = np.hstack((self.scaled_x, np.ones(self.scaled_x.shape)))

    def scale_dataset(self):
        return np.array([[self.scale(row[0], self.scaling_method), row[1]] for row in self.data])

    def scale(self, to_scale, method):
        if method == 'normalize':
            self.x_min = self.original_x.min()
            self.x_max = self.original_x.max()
            return (to_scale - self.x_min) / (self.x_max - self.x_min)
        elif method == 'standardize':
            self.x_mean = self.original_x.mean()
            self.x_std = self.original_x.std()
            return (to_scale - self.x_mean) / self.x_std

    def hypothesis(self):
        return self.x_oned.dot(self.theta)

    def cost_function(self):
        return 1 / (2 * len(self.y)) * np.sum((self.hypothesis() - self.y) ** 2)

    def gradient(self):
        return 1 / len(self.y) * self.x_oned.T.dot(self.hypothesis() - self.y)

    def fit(self):
        for i in range(0, self.n_iters):
            self.theta -= self.lr * self.gradient()
            self.cost_history[i] = self.cost_function()
            # print(i, self.cost_history[i])

    def coef_determination(self):
        u = ((self.y - self.hypothesis()) ** 2).sum()
        v = ((self.y - self.y.mean()) ** 2).sum()
        return 1 - u / v

    def visualize(self):
        # plot original data
        plt.title("Original data")
        plt.scatter(self.original_x, self.y)
        plt.xlabel(f'Mileage in km')
        plt.ylabel('Price in dollars')
        plt.show()

        # plot prediction line on training data
        plt.scatter(self.scaled_x, self.y, label='training data')
        plt.plot(self.scaled_x, self.hypothesis(), c='red')
        plt.title(f'Determinent coefficient : {self.coef_determination()}')
        plt.xlabel(f'Mileage in km ( {self.scaling_method}d )')
        plt.ylabel('Price in dollars')
        plt.legend(["prediction"])
        plt.show()

        # plot cost history
        plt.title("Cost history")
        plt.plot(range(self.n_iters), self.cost_history)
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
