import sys
import csv

from models import DataSet


def get_thetas():
    try:
        with open("data/thetas.csv", 'r') as csv_file:
            dict_val = csv.reader(csv_file, delimiter=",")
            return next(dict_val)
    except FileNotFoundError:
        print('Model not trained yet, thetas will be set to 0.')
        print('Consider running `python src/train.py` to train the model before predicting.\n')
        return [0, 0]


def get_user_input():
    mileage = None
    while mileage is None:
        try:
            mileage = int(input("Enter your car mileage in km: "))
        except ValueError:
            print("This is not a number !")
            continue
    return mileage


def main():
    dataset = DataSet('data/data.csv')
    [theta0, theta1, scaling] = get_thetas()

    mileage = dataset.scale(get_user_input(), scaling)
    prediction = float(theta1) + float(theta0) * mileage
    print(f'The approximated price of your car is {prediction} dollars.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
