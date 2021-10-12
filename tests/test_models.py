import pytest
from linear_regression.models import DataSet


def test_dataset_wrong_path(capfd):
    dataset = DataSet()
    with pytest.raises(SystemExit):
        dataset.load_csv('d')
    out, err = capfd.readouterr()
    assert err == "Wrong file or file path\n"
