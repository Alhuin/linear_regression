import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class DataSet:
    def __init__(self):
        self.data = None
        self.scaling_method = None
        self.x_min = None
        self.x_max = None
        self.x_mean = None
        self.x_std = None
        self.original_x = None
        self.y = None
        self.m = None
        self.scaled_x = None
        self.scaled_data = None

    def scale_dataset(self, scaling_method):
        """
            Scale the x-axis of the dataset
            :return: np.array: the scaled dataset
        """
        self.scaling_method = scaling_method
        self.scaled_data = np.array([
                [self.scale(row[0], scaling_method), row[1]]
                for row in self.data
            ])
        x = self.scaled_data[:, 0]
        self.scaled_x = x.reshape(x.shape[0], 1)

    def scale(self, to_scale, method):
        """
            Normalize or Standardize a variable
                - normalization: (x[i] - min(x)) / (max(x) - min(x))
                - standardization: (x[i] - mean(x)) / std(x)
            :param: float: to_scale:    the value to scale
            :param: str: method:        the scaling method
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

    def load_csv(self, csv_path):
        """
            Opens the csv_path and loads the data contained
                exit if csv_path is invalid

            :param csv_path:    the path containing the dataset as csv
        """
        try:
            self.data = np.loadtxt(
                open(csv_path),
                delimiter=",",
                skiprows=1
            )
        except FileNotFoundError:
            print("Wrong file or file path", file=sys.stderr)
            exit(0)
        except ValueError:
            print("Invalid data format", file=sys.stderr)
            exit(0)

        x = self.data[:, 0]
        y = self.data[:, 1]
        self.original_x = x.reshape(x.shape[0], 1)
        self.y = y.reshape(y.shape[0], 1)
        self.m = len(self.y)


class LinearRegression:
    def __init__(
            self,
            dataset,
            learning_rate=0.1,
            momentum=0,
            tolerance=0.00000001
    ):
        self.err_tolerance = tolerance
        self.dataset = dataset
        self.theta = np.zeros((2, 1))
        self.momentum = momentum
        self.iterations = 0
        self.lr = learning_rate
        self.theta_history = []
        self.cost_history = []
        self.x_oned = np.hstack((
            dataset.scaled_x,
            np.ones(dataset.scaled_x.shape)
        ))

    def hypothesis(self, theta):
        """"
            estimated_price(km) = theta1 * km + theta_0 =
            [[km[0], 1], [km[1], 1]...].[[theta_1], [theta0]] =
                [
                    [theta_1 * km[0] + theta_0 * 1],
                    [theta_1 * km[1] + theta_0 * 1], ...
                ]
            :return:
                the estimated values for each i in x as a matrix
        """
        return self.x_oned.dot(theta)

    def cost_function(self):
        """ (1 / 2 * m) * m∑i=1 (estimate(i) − real_i) ** 2 """
        return 1 / (2 * self.dataset.m) * np.sum(
            (self.hypothesis(self.theta) - self.dataset.y) ** 2
        )

    def gradient(self):
        """
            matrix representation of theta derivatives:

            x_oned = [[x[0], 1], [x[1], 1]...]
            x_oned_transposed = [[x[0], x[1]...], [1, 1...]]
            hypothesis(theta) - y =  [[diff0], [diff1]...]

            x_oned_transposed.dot(hypothesis(theta) - y) =
            [[x[0] * diff[0] + x[1] * diff1..], [1 * diff0 + 1 * diff1..]]

            :return: [[θ1_derivative], [θ0_derivative]] =
            [
                [1/m * {m-1∑i=0} (prixEstime(kilométrage[i]) − prix[i]) * x[i]]
                [1/m * {m-1∑i=0} prixEstime(kilométrage[i]) − prix[i]]
            ]
        """
        return 1 / self.dataset.m * self.x_oned.T.dot(
            self.hypothesis(self.theta) - self.dataset.y
        )

    def fit(self):
        """
            at each step, update thetas as:
            θ0 -= ratioDApprentissage ∗ 1/m * θ0_derivative
            θ1 -= ratioDApprentissage ∗ 1/m * θ1_derivative

            [[new_θ1], [new_θ0]] =
            [[θ1],[θ0]] - ratioDApprentissage * [[θ1_deriv],[θ0_deriv]]
        """
        self.theta_history.append(self.theta.copy())
        self.cost_history.append(self.cost_function())
        change = 0
        i = 0
        while True:
            # update thetas and apply momentum
            new_change = self.lr * self.gradient() + self.momentum * change
            self.theta -= new_change
            change = new_change
            # log
            self.theta_history.append(self.theta.copy())
            self.cost_history.append(self.cost_function())
            if i > 1 and abs(self.cost_history[-2] - self.cost_history[-1]) \
                    <= self.err_tolerance:
                self.iterations = i + 1
                break
            i += 1

    def coef_determination(self):
        """
            estimates the efficiency of the model, the closest to 1 the better
        :return:    A value between 0 and 1
        """
        u = ((self.dataset.y - self.hypothesis(self.theta)) ** 2).sum()
        v = ((self.dataset.y - self.dataset.y.mean()) ** 2).sum()
        return 1 - u / v


class Visualizer:
    def __init__(self, regressor):
        self.regressor = regressor

    def plot_cost_history(self):
        """
            Plots:
                - the training data         (blue points)
                - the current prediction    (red line)
                - the cost history          (blue line)
                - the current cost          (red point)
                - a slider to navigate threw iterations
        """
        fig = plt.figure(figsize=plt.figaspect(2.))
        prediction = fig.add_subplot(211)
        plt.subplots_adjust(
            left=0.15,
            bottom=0.1,
            right=0.8,
            top=0.8,
            wspace=0.4,
            hspace=0.8
        )

        # plot prediction line
        prediction.set_title('Prediction')
        prediction.scatter(
            self.regressor.dataset.scaled_x,
            self.regressor.dataset.y,
            label="Training data"
        )
        line, = prediction.plot(
            self.regressor.dataset.scaled_x,
            self.regressor.hypothesis(self.regressor.theta_history[0]),
            c='red',
            label="Current prediction"
        )
        prediction.set_ylabel('Price')
        prediction.set_ylim(0, 10000)
        prediction.legend()

        # plot cost history
        cost = fig.add_subplot(212)
        cost.set_title("Cost history")
        current_cost, = cost.plot(
            [0],
            self.regressor.cost_history[0],
            'ro',
            label="Current cost"
        )
        cost.plot(
            range(len(self.regressor.cost_history)),
            self.regressor.cost_history,
            label="Cost over iterations"
        )
        cost.legend()
        cost.set_xlabel('iterations')
        cost.set_ylabel('cost')

        # set slider to navigate threw steps
        ax_slide = plt.axes([0.15, 0.9, 0.65, 0.03])
        step = Slider(
            ax_slide,
            'step',
            0,
            self.regressor.iterations,
            valinit=0,
            valstep=1
        )

        def update_from_step(val):
            line.set_ydata(self.regressor.hypothesis(
                self.regressor.theta_history[val]
            ))
            fig.suptitle(f'current cost = {self.regressor.cost_history[val]}')
            current_cost.set_data(val, self.regressor.cost_history[val])

        def animate(i):
            step.set_val(i)

        animation = FuncAnimation(      # noqa
            fig,
            animate,
            frames=len(self.regressor.theta_history),
            interval=1,
            repeat=False
        )

        step.on_changed(update_from_step)
        plt.show()

    def plot_gradient_descent(self):
        """
            3D plot:
                - theta0            (x)
                - theta1            (y)
                - cost              (z)
                - cost history      (red line)
                - iterations        (blue points)
        """
        theta0 = [theta[1][0] for theta in self.regressor.theta_history]
        theta1 = [theta[0][0] for theta in self.regressor.theta_history]
        fig = plt.figure()
        fig.suptitle('Gradient descent')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            theta0,
            theta1,
            self.regressor.cost_history,
            'blue',
            label='steps'
        )
        ax.plot3D(
            theta0,
            theta1,
            self.regressor.cost_history,
            'red',
            label='gradient descent'
        )
        ax.set_xlabel('theta0')
        ax.set_ylabel('theta1')
        ax.set_zlabel('cost')
        fig.legend()
        plt.show()
