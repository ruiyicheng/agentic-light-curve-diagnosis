"""Unit tests for core functionality."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lightcurve_agent.core.models import LightCurve, Periodogram
from lightcurve_agent.core.data import (
    inspect_light_curve_file,
    load_light_curve,
    normalize_multiband_light_curve,
)
from lightcurve_agent.core.analysis.gls import phase_fold
from lightcurve_agent.interfaces.catalogs import _enrich_gaia_payload


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


class TestLightCurveLoading:
    """Tests for schema detection and multi-band handling."""

    def test_inspect_file_parses_header_and_source_metadata(self, tmp_path: Path):
        csv_path = tmp_path / "multiband.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "# name = RR Test",
                    "# ra = 291.047",
                    "# dec = 42.792",
                    "mjd,magnitude,magerr,filter",
                    "1.0,10.0,0.1,g",
                    "2.0,10.2,0.1,r",
                    "3.0,10.1,0.1,g",
                ]
            ),
            encoding="utf-8",
        )

        profile = inspect_light_curve_file(csv_path)

        assert profile.time_col == "mjd"
        assert profile.mag_col == "magnitude"
        assert profile.err_col == "magerr"
        assert profile.band_col == "filter"
        assert profile.source_name == "RR Test"
        assert profile.ra_deg == pytest.approx(291.047)
        assert profile.dec_deg == pytest.approx(42.792)
        assert profile.band_labels == ["g", "r"]

    def test_load_and_normalize_multiband_light_curve(self, tmp_path: Path):
        csv_path = tmp_path / "bands.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "time,mag,mag_err,filter",
                    "0.0,10.0,0.1,g",
                    "1.0,10.2,0.1,g",
                    "0.5,11.0,0.1,r",
                    "1.5,11.2,0.1,r",
                ]
            ),
            encoding="utf-8",
        )

        lc = load_light_curve(csv_path)
        normalized = normalize_multiband_light_curve(lc)

        assert lc.band_labels == ["g", "r"]
        assert normalized.band_labels == ["g", "r"]
        g_mask = normalized.band.astype(str) == "g"
        r_mask = normalized.band.astype(str) == "r"
        assert normalized.magnitude[g_mask].median() == pytest.approx(normalized.magnitude[r_mask].median())


class TestGaiaEnrichment:
    """Tests for Gaia-derived diagnosis helpers."""

    def test_enrich_gaia_payload_adds_galaxy_and_binary_hints(self):
        payload = _enrich_gaia_payload(
            {
                "teff_gspphot": 6400.0,
                "logg_gspphot": 4.1,
                "classprob_dsc_combmod_star": 0.1,
                "classprob_dsc_combmod_binarystar": 0.7,
                "classprob_dsc_combmod_galaxy": 0.8,
                "classprob_dsc_combmod_quasar": 0.05,
                "non_single_star": True,
            }
        )

        assert payload["stellar_parameters"]["teff_gspphot"] == 6400.0
        assert payload["stellar_parameters"]["logg_gspphot"] == 4.1
        assert payload["diagnosis_hints"]["is_probable_galaxy"] is True
        assert payload["diagnosis_hints"]["is_probable_binary"] is True
