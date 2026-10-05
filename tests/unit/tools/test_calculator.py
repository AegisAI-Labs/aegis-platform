import pytest

from aegis_platform.tools.calculator import calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2+3", "5"), ("10-4", "6"), ("3*3", "9"), ("8/2", "4.0")],
)
def test_calculator(expression: str, expected: str) -> None:
    assert calculator(expression) == expected


def test_calculator_rejects_invalid_expression() -> None:
    with pytest.raises(ValueError, match="Invalid calculator expression"):
        calculator("not arithmetic")
