"""VLM diagnosis tool for light curve plots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from lightcurve_agent.interfaces.vlm import get_vlm_provider

if TYPE_CHECKING:
    from lightcurve_agent.interfaces.vlm import VLMProvider


DEFAULT_DIAGNOSIS_PROMPT = """
This is a variable-star light curve plot (time vs brightness/magnitude).
Provide:
(1) Variability type hypotheses. Complete possible hypotheses should be listed within:
    Asteroid occultation, Planetary transits, EA, EB, EW, ELL, SXA, ACV, FKCOM, BY Dra,
    UV Ceti, RS CVn, RCB, FU, LBV, WR, GCAS, SN, ZAND, UG, Novae, RPHS, PV Tel,
    GW Vir, ZZ, Hot OB Supergiants, ACYG, BE, BCEP, SPB, SXPHE, DST, PMS DSCT,
    roAp, GDOR, RR, RV, CW, CEP, Mira, SR, irregulars, SARV, Non-variable, Unknown variable.
    Rule out impossible types conservatively based on magnitude range and morphology.

(2) For the whole light curve, describe morphology/trends/scatter/obvious long-term
    periodic features, and which hypotheses these support or rule out.

(3) Data-quality issues (outliers, systematics, saturation, cadence, incomplete uncertainty).

(4) Further inspection recommendations:
    (4.1) Whether GLS analysis is required for periodic features. Include a complete list
          of targets that should be diagnosed using GLS. Required only when periodic
          variable is a candidate.
    (4.2) Whether BLS analysis is required for periodic localized features. Include a
          complete list of targets for BLS diagnosis. Required only when EA or exoplanet
          is a candidate.
    (4.3) Whether zoom-in visualization is required for local morphological features.
          If so, specify 3 time ranges [t_start, t_end] in days to zoom in on.
          Only required when data points are too dense to see morphology clearly.

(5) Any irregular features.
"""


@tool
def diagnose_plot_with_vlm(
    image_path: str,
    question: str = DEFAULT_DIAGNOSIS_PROMPT,
) -> str:
    """Send a light curve plot to a vision-capable LLM for diagnosis.

    Args:
        image_path: Path to the PNG plot file
        question: Custom prompt for the VLM. Uses default if not provided.

    Returns:
        Diagnosis text from the VLM
    """
    try:
        print(f"Diagnosing plot: {image_path}")

        provider: VLMProvider = get_vlm_provider("openai")
        result = provider.analyze_image(image_path, question)

        print(f"Diagnosis complete")
        return result

    except Exception as e:
        return f"error: {str(e)}"
