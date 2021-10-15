import os
from linear_regression.predict import (
    get_user_input,
    import_globals,
    scale,
    main
)

std_in = 'builtins.input'
thetas = 'data/thetas.csv'


def gen_inputs(inputs):
    for in_ in inputs:
        yield in_


def test_get_user_input(monkeypatch, capfd):
    params = ('trois mille', ' ', '3000')
    expected = ('This is not a number !\n', 3000)
    inputs = gen_inputs(params)

    monkeypatch.setattr('builtins.input', lambda in_: next(inputs))
    ret = get_user_input()
    out, err = capfd.readouterr()
    assert err == expected[0] * (len(params) - 1)
    assert ret == expected[1]


def test_import_globals_no_thetas(capfd):
    if os.path.exists(thetas):
        os.remove(thetas)
    import_globals()
    out, err = capfd.readouterr()
    assert out == (
        'Model not trained yet, thetas will be set to 0.\n' +
        'Consider running `python src/train.py` to train the model.\n\n'
    )


def test_import_globals_w_thetas():
    f = open(thetas, "w+")
    f.write("%f,%f,%s,%f,%f" % (0, 1, 's', 2, 3))
    f.close()
    assert import_globals() == [
        '0.000000',
        '1.000000',
        's',
        '2.000000',
        '3.000000'
    ]


def test_scale_normalize():
    assert scale(2000, 'normalize', 0, 10000) == 0.2


def test_scale_standardize():
    assert scale(2000, 'standardize', 5000, 1000) == -3


def test_main_no_thetas(monkeypatch, capfd):
    if os.path.exists(thetas):
        os.remove(thetas)
    inputs = gen_inputs(('3000',))

    monkeypatch.setattr(std_in, lambda in_: next(inputs))
    main()
    out, err = capfd.readouterr()
    guess = out.split('\n')[-2]
    assert guess == 'The approximated price of your car is 0.0 dollars.'


def test_main_w_thetas(monkeypatch, capfd):
    f = open(thetas, "w+")
    f.write("%f,%f,%s,%f,%f" % (4, 5, 'standardize', 2, 3))
    f.close()
    inputs = gen_inputs(('3000',))

    monkeypatch.setattr(std_in, lambda in_: next(inputs))
    main()
    out, err = capfd.readouterr()
    assert out == (
            'The approximated price'
            ' of your car is 5000.67 dollars.\n'
    )
