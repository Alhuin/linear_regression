import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class LinearRegression:
    def __init__(self, path, learning_rate=0.1, n_iterations=500, scaling_method='normalize'):
        self.theta = np.zeros((2, 1))
        self.x_min = None
        self.x_max = None
        self.x_mean = None
        self.x_std = None
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.theta_history = [np.zeros((2, 1)) for _ in range(n_iterations)]
        self.cost_history = [[0] for _ in range(n_iterations)]
        self.scaling_method = scaling_method

        self.data = np.loadtxt(open(path), delimiter=",", skiprows=1)
        x = self.data[:, 0]
        y = self.data[:, 1]
        self.original_x = x.reshape(x.shape[0], 1)
        self.y = y.reshape(y.shape[0], 1)
        self.m = len(self.y)
        self.scaled_data = self.scale_dataset()
        x = self.scaled_data[:, 0]
        self.scaled_x = x.reshape(x.shape[0], 1)
        self.x_oned = np.hstack((self.scaled_x, np.ones(self.scaled_x.shape)))

    def scale_dataset(self):
        """
            Scale the x-axis of the dataset
            :return: np.array: the scaled dataset
        """
        return np.array([[self.scale(row[0], self.scaling_method), row[1]] for row in self.data])

    def scale(self, to_scale, method):
        """
            Normalize or Standardize a variable
                - normalization: (x[i] - min(x)) / (max(x) - min(x))
                - standardization: (x[i] - mean(x)) / std(x)
            :param: float: to_scale:    the value to scale
            :param: str: method:        the scaling methos to use (standardize or normalize)
            :return: float:             the scaled value
        """
        if method == 'normalize':
            self.x_min = self.original_x.min()
            self.x_max = self.original_x.max()
            return (to_scale - self.x_min) / (self.x_max - self.x_min)
        elif method == 'standardize':
            self.x_mean = self.original_x.mean()
            self.x_std = self.original_x.std()
            return (to_scale - self.x_mean) / self.x_std

    def hypothesis(self, theta):
        """"
            estimated_price(km) = theta1 * km + theta_0 = [km, 1].[theta_0, theta1]
            (a.b = a[0] * b[0] + a[1] * b[1] ...)
        """
        return self.x_oned.dot(theta)

    def cost_function(self):
        """ (1 / 2 * m) * m∑i=1 (estimate(i) − real_i) ** 2 """
        return 1 / (2 * self.m) * np.sum((self.hypothesis(self.theta) - self.y) ** 2)

    def gradient(self):
        return 1 / self.m * self.x_oned.T.dot(self.hypothesis(self.theta) - self.y)

    def fit(self):
        self.theta_history[0] = self.theta
        self.cost_history[0] = self.cost_function()
        for i in range(0, self.n_iters):
            self.theta -= self.lr * self.gradient()     # update theta0 & theta_1
            self.theta_history[i] = self.theta.copy()
            self.cost_history[i] = self.cost_function()

    def coef_determination(self):
        u = ((self.y - self.hypothesis(self.theta)) ** 2).sum()
        v = ((self.y - self.y.mean()) ** 2).sum()
        return 1 - u / v

    def visualize(self):
        fig, (prediction, cost) = plt.subplots(2)
        plt.subplots_adjust(left=0.15,
                            bottom=0.1,
                            right=0.9,
                            top=0.8,
                            wspace=0.4,
                            hspace=0.8)

        # plot prediction line
        prediction.set_title('Prediction')
        prediction.scatter(self.scaled_x, self.y, label="Training data")
        line, = prediction.plot(self.scaled_x, self.hypothesis(self.theta_history[0]), c='red', label="Current prediction")
        prediction.set_xlabel(f'Mileage in km ( {self.scaling_method}d )')
        prediction.set_ylabel('Price in dollars')
        prediction.set_ylim(0, 10000)
        prediction.legend()

        # plot cost history
        cost.set_title("Cost history")
        current_cost, = cost.plot([0], self.cost_history[0], 'ro', label="Current cost")
        cost.plot(range(self.n_iters), self.cost_history, label="Cost over iterations")
        cost.legend()
        cost.set_xlabel('iterations')
        cost.set_ylabel('cost')

        def animate(i):
            fig.suptitle(f' step {i}')
            line.set_ydata(self.hypothesis(self.theta_history[i]))
            current_cost.set_data(i, self.cost_history[i])

        animation = FuncAnimation(
            fig,
            animate,
            frames=len(self.theta_history),
            interval=5000 / self.n_iters,
            repeat=False
        )
        plt.show()
