import sys
import csv


def import_globals():
    try:
        with open("data/thetas.csv", 'r') as csv_file:
            dict_val = csv.reader(csv_file, delimiter=",")
            return next(dict_val)
    except FileNotFoundError:
        print('Model not trained yet, thetas will be set to 0.')
        print('Consider running `python src/train.py` to train the model before predicting.\n')
        return [0, 0, None, None, None]


def get_user_input():
    mileage = None
    while mileage is None:
        try:
            mileage = int(input("Enter your car mileage in km: "))
        except ValueError:
            print("This is not a number !")
            continue
    return mileage


def scale(to_scale, method, param_1, param_2):
    if method == 'normalize':
        x_min, x_max = \
            float(param_1), float(param_2)
        return (to_scale - x_min) / (x_max - x_min)
    elif method == 'standardize':
        x_mean, x_std = float(param_1), float(param_2)
        return (to_scale - x_mean) / x_std


def main():
    [theta0, theta1, scaling_method, scaling_param_1, scaling_param_2] = import_globals()

    mileage = scale(get_user_input(), scaling_method, scaling_param_1, scaling_param_2)
    prediction = float(theta1) + float(theta0) * mileage
    print(f'The approximated price of your car is {prediction} dollars.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
