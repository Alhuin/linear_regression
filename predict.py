import sys
import csv


def get_thetas():
    try:
        with open("thetas.csv", 'r') as csv_file:
            dict_val = csv.reader(csv_file, delimiter=",")
            return next(dict_val)
    except FileNotFoundError:
        print('Model not trained yet, thetas will be set to 0.')
        print('Consider running `python train.py` to train the model before predicting.\n')
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
    [theta0, theta1] = get_thetas()
    mileage = get_user_input()
    prediction = float(theta0) + float(theta1) * mileage
    print(f'The approximated price of your car is {prediction} dollars.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted, exiting.')
