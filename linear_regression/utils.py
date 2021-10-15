import argparse
import sys
import csv
import numpy as np
from linear_regression.models import LinearRegression


def parse(args):
    """
        Parse flags and arguments from sys.argv
        :return:
            str: scaling_method:    the scaling method used to fit the dataset
            boolean: visualize:     if plotting is needed
    """
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Linear regression trainer.'
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        help='manually set the learning rate (between 0 and 1)',
        default=0.1
    )
    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        help='manually set momentum value (between 0 and 1)',
        default=0
    )
    parser.add_argument(
        '-t',
        '--tolerance',
        type=float,
        help='tolerance in error reduction for the stop condition.',
        default=0.00001
    )
    parser.add_argument(
        '-g',
        '--grid-search',
        action='store_true',
        help=(
            'perform a grid search '
            'to find best lr and momentum (will override both)'
        ),
        default=False
    )
    parser.add_argument(
        '-s',
        '--scale',
        type=str,
        help='the method used to scale the dataset (normalize or standardize)',
        default="standardize"
    )
    parser.add_argument(
        '-v',
        '--visualize',
        action='store_true',
        help='visualize data',
        default=False
    )
    return parser.parse_args(args)


def export_globals(regressor, dataset):
    """
        Write the global variables to a csv file to be imported in predict.py
            globals:
                - theta0
                - theta1
                - scaling_method
                - scaling_param1
                - scaling_param2
        :param: regressor: the trained LinearRegression object
        :return: None
    """
    f = open("data/thetas.csv", "w+")
    if dataset.scaling_method == 'normalize':
        scaling_param1 = dataset.x_min
        scaling_param2 = dataset.x_max
    else:
        scaling_param1 = dataset.x_mean
        scaling_param2 = dataset.x_std
    f.write("%f,%f,%s,%f,%f" % (
        regressor.theta[1],
        regressor.theta[0],
        dataset.scaling_method,
        scaling_param1,
        scaling_param2
    )
            )
    f.close()


def grid_search(dataset, lr_bounds, m_bounds):
    """

    :param dataset:     the scaled DataSet object to train on
    :param lr_bounds:   the learning_rate min and max values as [min, max]
    :param m_bounds:    the momentum min and max values as [min, max]
    :return:            the best learning rate, the best momentum and the minimal error     # noqa
    """
    # generate evenly spaced lists of parameters
    lr_range = np.arange(lr_bounds[0], lr_bounds[1], 0.1)
    m_range = np.arange(m_bounds[0], m_bounds[1], 0.1)
    learning_rates = []
    momentums = []
    iterations = []
    costs = []
    for lr_value in lr_range:
        learning_rates_row = []
        momentums_row = []
        iterations_row = []
        costs_row = []
        for momentum_value in m_range:
            regressor = LinearRegression(
                dataset,
                learning_rate=lr_value,
                momentum=momentum_value
            )
            regressor.fit()
            learning_rates_row.append(lr_value)
            momentums_row.append(momentum_value)
            iterations_row.append(regressor.iterations)
            costs_row.append(regressor.cost_function())
        learning_rates.append(learning_rates_row)
        momentums.append(momentums_row)
        iterations.append(iterations_row)
        costs.append(costs_row)

    learning_rates = np.array(learning_rates)
    momentums = np.array(momentums)
    iterations = np.array(iterations)
    costs = np.array(costs)
    pos_min_costs = np.argwhere(costs == np.min(costs))
    iterations = np.array([
        iterations[pos[0], pos[1]]
        for pos in pos_min_costs]
    )
    min_i = np.min(iterations)
    pos_min_i = np.argwhere(iterations == np.min(iterations))[0][0]

    best_lr = learning_rates[
        pos_min_costs[pos_min_i][0],
        pos_min_costs[pos_min_i][1]
    ]
    best_momentum = momentums[
        pos_min_costs[pos_min_i][0],
        pos_min_costs[pos_min_i][1]
    ]
    return best_lr, best_momentum, min_i


def import_globals():
    """
        Import the global variables from the csv file created in predict.py
        :return:
            [theta0, theta1, scaling_method, scaling_param1, scaling_param2]
    """
    try:
        with open("data/thetas.csv", 'r') as csv_file:
            dict_val = csv.reader(csv_file, delimiter=",")
            return next(dict_val)
    except FileNotFoundError:
        print('Model not trained yet, thetas will be set to 0.')
        print('Consider running `python src/train.py` to train the model.\n')
        return [0, 0, None, None, None]


def get_user_input():
    """
        Prompt user for a mileage
        :return: int: mileage
    """
    mileage = None
    while mileage is None:
        try:
            mileage = int(input("Enter your car mileage in km: "))
        except ValueError:
            print("This is not a number !", file=sys.stderr)
            continue
    return mileage


def scale(to_scale, method, param_1, param_2):
    """
        Scale a number with the parameters given used by the regressor

        :param to_scale:    the number to be scaled
        :param method:      the scaling method (standardize or normalize)
        :param param_1:     the first scaling parameter
        :param param_2:     the second scaling parameter

        :return: the scaled variable
    """
    if method == 'normalize':
        x_min, x_max = \
            float(param_1), float(param_2)
        return (to_scale - x_min) / (x_max - x_min)
    elif method == 'standardize':
        x_mean, x_std = float(param_1), float(param_2)
        return (to_scale - x_mean) / x_std
