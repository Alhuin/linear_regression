import numpy as np
# import matplotlib.pyplot as plt


class DataSet:
    def __init__(self, path):
        self.data = np.loadtxt(open(path), delimiter=",", skiprows=1)
        x = self.data[:, 0]
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.scaled_data = None

    def scale_data(self, type):
        self.scaled_data = np.array([
            [self.scale(row[0], type), row[1]] for row in self.data
        ])

    def scale(self, input, type):
        if type == 'normalize':
            return self.normalize(input)
        elif type == 'standardize':
            return self.standardize(input)

    def standardize(self, input):
        return (input - self.x_mean) / self.x_std

    def normalize(self, input):
        return (input - self.x_min) / (self.x_max - self.x_min)


class LinearRegression:
    def __init__(self, dataset, learning_rate=0.1, n_iterations=500):
        x = dataset[:, 0]
        y = dataset[:, 1]
        self.x = x.reshape(x.shape[0], 1)
        self.y = y.reshape(y.shape[0], 1)
        self.x_oned = np.hstack((self.x, np.ones(self.x.shape)))

        self.theta = np.zeros((2, 1))
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.cost_history = np.zeros(n_iterations)

    def model(self):
        return self.x_oned.dot(self.theta)

    def cost_function(self):
        return 1 / (2 * len(self.y)) * np.sum((self.model() - self.y) ** 2)

    def gradient(self):
        return 1 / len(self.y) * self.x_oned.T.dot(self.model() - self.y)

    def fit(self):
        for i in range(0, self.n_iters):
            self.theta -= self.lr * self.gradient()
            self.cost_history[i] = self.cost_function()
        #     plt.subplot(211)
        #     plt.scatter(self.x, self.y, c='lightblue')
        #     plt.xlabel("mileage (kilometers)")
        #     plt.ylabel("price")
        #     if i % 99 == 0:
        #         c = plt.plot(self.x, self.model(), c='red')
        #         plt.legend(["prediction"])
        #         plt.pause(0.2)
        #         for handler in c:
        #             handler.remove()
        #
        # plt.plot(self.x, self.model(), c='red')
        # plt.show()

    def coef_determination(self):
        u = ((self.y - self.model()) ** 2).sum()
        v = ((self.y - self.y.mean()) ** 2).sum()
        return 1 - u / v
