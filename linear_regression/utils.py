import argparse
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
        prog='src/train.py',
        description='Linear regression trainer.'
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        help='manually set the learning rate',
        default=0.01
    )
    parser.add_argument(
        '-i',
        '--iterations',
        type=int,
        help='manually set the number of iterations',
        default=150
    )
    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        help='manually set momentum value',
        default=0
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
    costs = []
    for lr_value in lr_range:
        learning_rates_row = []
        momentums_row = []
        costs_row = []
        for momentum_value in m_range:
            regressor = LinearRegression(
                dataset,
                learning_rate=lr_value,
                momentum=momentum_value
            )
            regressor.fit()
            cost = regressor.cost_function()
            learning_rates_row.append(lr_value)
            momentums_row.append(momentum_value)
            costs_row.append(cost)
        learning_rates.append(learning_rates_row)
        momentums.append(momentums_row)
        costs.append(costs_row)

    learning_rates = np.array(learning_rates)
    momentums = np.array(momentums)
    costs = np.array(costs)
    min_cost = np.min(costs)
    pos_min_cost = np.argwhere(costs == np.min(costs))[0]

    best_lr = learning_rates[pos_min_cost[0], pos_min_cost[1]]
    best_momentum = momentums[pos_min_cost[0], pos_min_cost[1]]
    return best_lr, best_momentum, min_cost
