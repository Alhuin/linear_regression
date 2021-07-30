import sys

import matplotlib.pyplot as plt
from models import LinearRegression, DataSet


def main():
    scaling = 'normalize'
    if len(sys.argv) > 1:
        if sys.argv[1] in ['standardize', 'normalize']:
            scaling = sys.argv[1]
    dataset = DataSet('data/data.csv')
    dataset.scale_data(type=scaling)
    regressor = LinearRegression(dataset.scaled_data)
    regressor.fit()

    f = open("data/thetas.csv", "w+")
    f.write("%f, %f, %s" % (regressor.theta[0], regressor.theta[1], scaling))
    f.close()

    plt.subplot(211)
    plt.scatter(regressor.x, regressor.y)
    plt.plot(regressor.x, regressor.model(), c='red')
    plt.title(f'Determinent coefficient : {regressor.coef_determination()}')
    plt.xlabel("mileage (kilometers)")
    plt.ylabel("price")
    plt.legend(["prediction"])

    plt.subplot(212)
    plt.plot(range(regressor.n_iters), regressor.cost_history)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.subplots_adjust(wspace=2, hspace=0.4)
    plt.show()


if __name__ == "__main__":
    main()
