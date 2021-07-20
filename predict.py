import sys
import csv


def get_thetas():
    thetas = []

    try:
        with open("thetas.csv", 'r') as csv_file:
            dict_val = csv.reader(csv_file, delimiter=",")
            thetas = next(dict_val)
    except FileNotFoundError:
        print('Model not trained yet, thetas will be set to 0.')
        print('Consider running `python train.py` to train the model before predicting.\n')
        return [0, 0]
    return thetas


def get_user_input():
    km = None
    while km is None:
        try:
            km = int(input("Enter your car mileage in km: "))
        except ValueError:
            print("This is not a number !")
            continue
    return km


def main():
    [theta0, theta1] = get_thetas()
    # print(f'theta0 = {theta0}')
    # print(f'theta1 = {theta1}\n')
    mileage = get_user_input()
    prediction = float(theta0) + float(theta1) * mileage
    print(f'The approximated price of your car is {prediction} dollars.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted, exiting.')
        sys.exit(0)
