import numpy as np
import pandas as pd
import pytest

from rates_backtester import add_nelson_siegel_factors, fit_nelson_siegel_curve, nelson_siegel_curve


def test_fixed_decay_fit_recovers_known_factors():
    tenors = np.asarray([1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    yields = nelson_siegel_curve(tenors, 4.0, -1.5, 2.0)
    fit = fit_nelson_siegel_curve(tenors, yields)
    assert fit.level == pytest.approx(4.0)
    assert fit.slope == pytest.approx(-1.5)
    assert fit.curvature == pytest.approx(2.0)
    assert fit.rmse_bp == pytest.approx(0.0, abs=1e-10)


def test_factors_can_be_fit_from_existing_four_tenors():
    tenors = np.asarray([2.0, 5.0, 10.0, 30.0])
    yields = nelson_siegel_curve(tenors, 4.0, -1.5, 2.0)
    frame = pd.DataFrame(
        [dict(zip(("2Y", "5Y", "10Y", "30Y"), yields))],
        index=pd.to_datetime(["2024-01-02"]),
    )
    enriched = add_nelson_siegel_factors(frame)
    assert enriched.loc[frame.index[0], "ns_level"] == pytest.approx(4.0)
    assert enriched.loc[frame.index[0], "ns_tenor_count"] == 4
