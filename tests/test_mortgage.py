import numpy as np

from buy_vs_rent.mortgage import amortize_year, monthly_payment


def test_monthly_payment_matches_standard_formula():
    payment = monthly_payment(
        np.array([100_000.0]), np.array([0.06]), np.array([360])
    )[0]
    assert payment == pytest.approx(599.5505, rel=1e-6)


def test_thirty_year_loan_amortizes_to_zero():
    balance = np.array([950_000.0])
    rate = np.array([0.0666])
    months = np.array([360])
    for _ in range(30):
        _, _, balance, months = amortize_year(balance, rate, months)
    assert balance[0] == pytest.approx(0.0, abs=0.01)
    assert months[0] == 0


import pytest

