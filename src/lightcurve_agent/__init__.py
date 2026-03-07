"""Light Curve Agent - AI-powered astronomical light curve analysis."""

__version__ = "0.1.0"

from lightcurve_agent.config import get_settings, Settings
from lightcurve_agent.agent import create_diagnosis_agent

__all__ = ["get_settings", "Settings", "create_diagnosis_agent"]
