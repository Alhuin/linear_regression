import sys
import csv


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
        print('Consider running `python src/train.py` to train the model before predicting.\n')
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
            print("This is not a number !")
            continue
    return mileage


def scale(to_scale, method, param_1, param_2):
    """
        Scale a number with the parameters given used by the regressor

        :param to_scale:    the number to be scaled
        :param method:      the scaling method (standardize or normalize)
        :param param_1:     the first scaling parameter (mean(x) for standardization, min(x) for normalization)
        :param param_2:     the second scaling parameter (std(x) for standardization, max(x) for normalization)

        :return: the scaled variable
    """
    if method == 'normalize':
        x_min, x_max = \
            float(param_1), float(param_2)
        return (to_scale - x_min) / (x_max - x_min)
    elif method == 'standardize':
        x_mean, x_std = float(param_1), float(param_2)
        return (to_scale - x_mean) / x_std


def main():

    # import globals from the csv
    [theta0, theta1, scaling_method, scaling_param_1, scaling_param_2] = import_globals()

    # get user input and scale the value using the regressor parameters
    mileage = get_user_input()
    if scaling_method is not None:
        mileage = scale(mileage, scaling_method, scaling_param_1, scaling_param_2)

    # predict with the regressor thetas (y = ax + b) and print the results
    prediction = float(theta0) + float(theta1) * mileage
    print(f'The approximated price of your car is {prediction} dollars.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
