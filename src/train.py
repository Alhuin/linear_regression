import sys
from models import LinearRegression, Visualizer


def parse():
    """
        Parse flags and arguments from sys.argv
        :return:
            str: scaling_method:    the scaling method used to train the dataset ( normalize or standardize )
            boolean: visualize:     if plotting is needed
    """
    scaling_method = 'standardize'
    visualize = False
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-h" or arg == "--help":
            print('Usage: `\n'
                  '\tpython3 src/train.py {options}\n\n'
                  '\t-h|--help:\tshow usage\n'
                  '\t-s|--scale {normalize|standardize}: set scaling method (default is standardize)\n'
                  '\t-v|--visualize {opc}: plot data (default is opc)\n'
                  '\t\to: original data\n'
                  '\t\tp: prediction on scaled data\n'
                  '\t\tc: cost function over iterations\n')
            sys.exit(0)
        elif arg == "-s" or arg == "--scale":
            if len(sys.argv) > i + 1 and sys.argv[i+1] in ['normalize', 'standardize']:
                scaling_method = sys.argv[i+1]
            else:
                print('Scaling option needs a valid method.\n'
                      'Run `python3 src/train.py {-h|--help}` to print usage.')
                sys.exit(0)
        elif arg == "-v" or arg == "--visualize":
            visualize = True
        i += 1
    return scaling_method, visualize


def export_globals(regressor):
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
    f.write("%f,%f,%s,%f,%f" % (
        regressor.theta[1],
        regressor.theta[0],
        regressor.scaling_method,
        regressor.x_min if regressor.scaling_method == 'normalize' else regressor.x_mean,
        regressor.x_max if regressor.scaling_method == 'normalize' else regressor.x_std
    ))
    f.close()


def main():
    # These are the optimized parameters for learning_rate and iters, you can play with them here !
    best_fits = {
        'standardize': {'lr': 0.1, 'iters': 100},
        'normalize': {'lr': 0.1, 'iters': 1000}
    }

    # Parse flags and arguments
    scaling_method, visualize = parse()

    # Init regressor
    regressor = LinearRegression(
        path='data/data.csv',
        learning_rate=best_fits[scaling_method]['lr'],
        n_iterations=best_fits[scaling_method]['iters'],
        scaling_method=scaling_method
    )

    # Train model
    regressor.fit()

    # Export globals to import in predict.py
    export_globals(regressor)

    # Plot graphs
    if visualize:
        visualizer = Visualizer(regressor)
        visualizer.plot_cost_history()
        visualizer.plot_gradient_descent()


if __name__ == "__main__":
    main()
