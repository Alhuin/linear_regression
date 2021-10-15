import sys
from linear_regression.utils import parse, export_globals, grid_search
from linear_regression.models import (
    LinearRegression,
    Visualizer,
    DataSet
)


def main(args=None):
    # Parse flags and arguments

    if args is None:
        # get arguments from parser
        args = parse(sys.argv[1:])
        lr = args.learning_rate
        momentum = args.momentum
        g_search = args.grid_search
        visualize = args.visualize
        scaling_method = args.scale
        tolerance = args.tolerance
    else:
        # get arguments from call (testing purpose)
        [scaling_method, visualize, lr, momentum, g_search, tolerance] = args

    # Create, load and scale dataset
    dataset = DataSet()
    dataset.load_csv('data/data.csv')
    dataset.scale_dataset(scaling_method)

    # Find the best hyperparameters learning_rate and momentum
    if g_search is True:
        print(
            'Grid Search enabled, '
            'learning_rate and momentum values will be overwritten'
        )
        lr, momentum, iterations = grid_search(dataset, [0, 1], [0, 1])

    print(f'parameters lr: {lr}, momentum: {momentum}, tolerance: {tolerance}')
    # Train the model with the best parameters
    regressor = LinearRegression(
        dataset=dataset,
        learning_rate=lr,
        momentum=momentum,
        tolerance=tolerance
    )
    regressor.fit()
    print(f'Found fit under tolerance at turn {regressor.iterations}')

    # Export globals to import in predict.py
    export_globals(regressor, dataset)
    print(f'coefficient of determination : '
          f'{regressor.coef_determination()}\n'
          f'minimum error : '
          f'{regressor.cost_function()}')

    # Plot graphs
    if visualize:
        visualizer = Visualizer(regressor)
        visualizer.plot_cost_history()
        visualizer.plot_gradient_descent()


if __name__ == "__main__":                          # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
