import csv
import os
import pytest

from linear_regression.train import parse, export_globals, grid_search, main
from linear_regression.models import LinearRegression, DataSet

thetas = "data/thetas.csv"


def test_parse_no_args():
    args = parse([])
    assert args.scale == 'standardize'
    assert args.visualize is False


def test_parse_normalize():
    args = parse(['-s', 'normalize'])
    assert args.scale == 'normalize'
    assert args.visualize is False


def test_parse_standardize():
    args = parse(['-s', 'standardize'])
    assert args.scale == 'standardize'
    assert args.visualize is False


def test_parse_visualize():
    args = parse(['-v'])
    assert args.scale == 'standardize'
    assert args.visualize is True


def test_parser_print_help_and_exit(capfd):
    with pytest.raises(SystemExit):
        parse(['-h'])
    out, err = capfd.readouterr()
    assert out == (
        "usage: src/train.py [-h] [-lr LEARNING_RATE] "
        "[-i ITERATIONS] [-m MOMENTUM]\n"
        "                    [-g] [-s SCALE] [-v]\n\n"
        "Linear regression trainer.\n\n"
        "optional arguments:\n"
        "  -h, --help            show this help message and exit\n"
        "  -lr LEARNING_RATE, --learning-rate LEARNING_RATE\n"
        "                        manually set the learning rate\n"
        "  -i ITERATIONS, --iterations ITERATIONS\n"
        "                        manually set the number of iterations\n"
        "  -m MOMENTUM, --momentum MOMENTUM\n"
        "                        manually set momentum value\n"
        "  -g, --grid-search     perform a grid search "
        "to find best lr and momentum\n"
        "                        (will override both)\n"
        "  -s SCALE, --scale SCALE\n"
        "                        the method used to scale "
        "the dataset (normalize or\n"
        "                        standardize)\n"
        "  -v, --visualize       visualize data\n"
    )


def test_parser_unrecognized_aguments(capfd):
    with pytest.raises(SystemExit):
        parse(['-q'])
    out, err = capfd.readouterr()
    assert err == (
        "usage: src/train.py [-h] [-lr LEARNING_RATE] "
        "[-i ITERATIONS] [-m MOMENTUM]\n"
        "                    [-g] [-s SCALE] [-v]\n"
        "src/train.py: error: unrecognized arguments: -q\n"
    )


def test_export_globals_standardize():
    if os.path.exists(thetas):
        os.remove(thetas)

    dataset = DataSet()
    dataset.load_csv('data/data.csv')
    dataset.scale_dataset('standardize')
    regressor = LinearRegression(
        dataset,
        learning_rate=0.01,
        n_iters=100,
    )
    export_globals(regressor, dataset)
    with open(thetas, 'r') as csv_file:
        dict_val = csv.reader(csv_file, delimiter=",")
        assert next(dict_val) == [
            '0.000000',
            '0.000000',
            'standardize',
            '101066.250000',
            '51565.189911'
        ]


def test_export_globals_normalize():
    if os.path.exists(thetas):
        os.remove(thetas)

    dataset = DataSet()
    dataset.load_csv('data/data.csv')
    dataset.scale_dataset('normalize')
    regressor = LinearRegression(
        dataset,
        learning_rate=0.01,
        n_iters=100,
    )
    export_globals(regressor, dataset)
    with open(thetas, 'r') as csv_file:
        dict_val = csv.reader(csv_file, delimiter=",")
        assert next(dict_val) == [
            '0.000000',
            '0.000000',
            'normalize',
            '22899.000000',
            '240000.000000'
        ]


def test_grid_search_on_normalized_data():
    dataset = DataSet()
    dataset.load_csv('data/data.csv')
    dataset.scale_dataset('normalize')
    lr, momentum, min_err = grid_search(dataset, [0, 1], [0, 1])
    assert lr == 0.6000000000000001
    assert momentum == 0.7000000000000001
    assert min_err == 222822.62253563665


def test_grid_search_on_standardized_data():
    dataset = DataSet()
    dataset.load_csv('data/data.csv')
    dataset.scale_dataset('standardize')
    lr, momentum, min_err = grid_search(dataset, [0, 1], [0, 1])
    assert lr == 0.1
    assert momentum == 0.4
    assert min_err == 222822.62253563665


def test_main_no_args():
    if os.path.exists(thetas):
        os.remove(thetas)
    main()
    assert os.path.exists(thetas) is True


def test_main_w_grid_search(capfd):
    main(['standardize', False, 0.1, 0, True, 100])
    out, err = capfd.readouterr()
    assert (
                "Grid Search enabled, "
                "learning_rate and momentum values will be overwritten"
           ) in out


def test_main_no_vis():
    if os.path.exists(thetas):
        os.remove(thetas)
    main(['standardize', False, 0.1, 0, False, 100])
    assert os.path.exists(thetas) is True


def test_main_w_vis():
    if os.path.exists(thetas):
        os.remove(thetas)
    main(['standardize', True, 0.1, 0, False, 100])
    assert os.path.exists(thetas) is True
