from rates_backtester.trades import normalize_gross_weights


def test_normalize_gross_weights():
    weights = normalize_gross_weights((2.0, -1.0), 1.0)
    assert sum(abs(weight) for weight in weights) == 1.0
