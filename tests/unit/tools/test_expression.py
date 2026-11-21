import pytest

from aegis_platform.tools.expression import extract_expression, has_calculator_keyword


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("calculate 2+3", "2+3"),
        ("calculator 7 * 9", "7 * 9"),
        ("Please calculate (1+2)*3?", "(1+2)*3"),
        ("calculate: 10 / 4 =", "10 / 4"),
        ("add 2 and 3", "2 + 3"),
        ("Please ADD 2 to 3.5", "2 + 3.5"),
        ("subtract 2 from 5", "5 - 2"),
        ("Please MULTIPLY 2 by 3", "2 * 3"),
        ("multiply -4 times 2", "-4 * 2"),
        ("divide 6 by 2", "6 / 2"),
    ],
)
def test_extract_expression(message: str, expected: str) -> None:
    assert extract_expression(message) == expected


@pytest.mark.parametrize(
    "message",
    ["calculate invalid", "calculate the area of a circle", "add some salt", "Hello", "calculate"],
)
def test_no_expression(message: str) -> None:
    assert extract_expression(message) is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("calculate 2+3", True),
        ("Please MULTIPLY 2 by 3", True),
        ("what is my address?", False),
        ("add padding to the div", True),
        ("padding and addresses", False),
        ("calc:2+3", False),
    ],
)
def test_whole_word_keywords(message: str, expected: bool) -> None:
    assert has_calculator_keyword(message) is expected
