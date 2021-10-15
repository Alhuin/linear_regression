import sys
from linear_regression.utils import import_globals, get_user_input, scale


def main():
    # import globals from the csv
    [
        theta0,
        theta1,
        scaling_method,
        scaling_param_1,
        scaling_param_2
    ] = import_globals()

    # get user input and scale the value using the regressor parameters
    mileage = get_user_input()
    if scaling_method is not None:
        mileage = scale(mileage, scaling_method,
                        scaling_param_1, scaling_param_2)

    # predict with the regressor thetas (y = ax + b) and print the results
    prediction = float(theta0) + float(theta1) * mileage
    print(f'The approximated price of your car is '
          f'{round(prediction, 2)} dollars.'
          )


if __name__ == '__main__':                      # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
