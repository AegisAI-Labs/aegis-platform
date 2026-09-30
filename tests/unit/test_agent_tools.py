from aegis_platform.agent.tools import calculator


def test_calculator_addition():
    assert calculator("2+3") == 5


def test_calculator_subtraction():
    assert calculator("10-4") == 6


def test_calculator_multiplication():
    assert calculator("3*3") == 9


def test_calculator_division():
    assert calculator("8/2") == 4.0
