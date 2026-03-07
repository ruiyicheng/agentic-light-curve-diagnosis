"""Unit tests for core functionality."""

import numpy as np
import pandas as pd
import pytest

from lightcurve_agent.core.models import LightCurve, Periodogram
from lightcurve_agent.core.data import load_light_curve
from lightcurve_agent.core.analysis.gls import phase_fold


class TestLightCurve:
    """Tests for LightCurve model."""

    def test_create_lightcurve(self):
        """Test basic LightCurve creation."""
        lc = LightCurve(
            time=pd.Series([0.0, 1.0, 2.0]),
            magnitude=pd.Series([10.0, 10.5, 10.2]),
            error=pd.Series([0.1, 0.1, 0.1]),
        )
        assert len(lc.time) == 3
        assert lc.scale == "mag"  # default

    def test_lightcurve_validation(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError):
            LightCurve(
                time=pd.Series([0.0, 1.0]),
                magnitude=pd.Series([10.0, 10.5, 10.2]),
            )

    def test_masked(self):
        """Test masking functionality."""
        lc = LightCurve(
            time=pd.Series([0.0, 1.0, 2.0, 3.0]),
            magnitude=pd.Series([10.0, 10.5, 10.2, 10.8]),
        )
        mask = np.array([True, True, False, False])
        masked = lc.masked(mask)
        assert len(masked.time) == 2


class TestPhaseFold:
    """Tests for phase folding."""

    def test_phase_fold_basic(self):
        """Test basic phase folding."""
        lc = LightCurve(
            time=pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]),
            magnitude=pd.Series([10.0, 10.5, 10.2, 10.8, 10.1]),
        )
        folded = phase_fold(lc, period=2.0)
        # All times should be in [0, 2)
        assert all((folded >= 0) & (folded < 2.0))

    def test_phase_fold_periodicity(self):
        """Test that phase folding is periodic."""
        lc = LightCurve(
            time=pd.Series([0.0, 2.0, 4.0]),  # 2 periods apart
            magnitude=pd.Series([10.0, 10.5, 10.2]),
        )
        folded = phase_fold(lc, period=2.0)
        # All should fold to same phase
        np.testing.assert_allclose(folded.values, [0.0, 0.0, 0.0], atol=1e-10)


class TestPeriodogram:
    """Tests for Periodogram model."""

    def test_create_periodogram(self):
        """Test Periodogram creation."""
        p = Periodogram(
            grid=pd.Series([1.0, 2.0, 3.0]),
            power=pd.Series([0.1, 0.5, 0.2]),
            best_value=2.0,
            best_power=0.5,
        )
        assert p.best_value == 2.0
        assert p.best_power == 0.5

    def test_periodogram_validation(self):
        """Test that mismatched grid/power raises error."""
        with pytest.raises(ValueError):
            Periodogram(
                grid=pd.Series([1.0, 2.0, 3.0]),
                power=pd.Series([0.1, 0.5]),  # Wrong length
            )
