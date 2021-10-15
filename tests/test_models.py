import pytest
from linear_regression.models import DataSet


def test_dataset_wrong_path(capfd):
    dataset = DataSet()
    with pytest.raises(SystemExit):
        dataset.load_csv('d')
    out, err = capfd.readouterr()
    assert err == "Wrong file or file path\n"


def test_dataset_wrong_format(capfd):
    dataset = DataSet()
    with pytest.raises(SystemExit):
        dataset.load_csv('data/data_wrong_format.csv')
    out, err = capfd.readouterr()
    assert err == "Invalid data format\n"
