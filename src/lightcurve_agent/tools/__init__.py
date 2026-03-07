"""LangChain tool definitions for the light curve agent.

These wrap the core functionality in @tool decorators for agent use.
"""

from lightcurve_agent.tools.plot import plot_light_curve_csv
from lightcurve_agent.tools.diagnose import diagnose_plot_with_vlm
from lightcurve_agent.tools.periodogram import obtain_GLS, obtain_BLS, analysis_LS
from lightcurve_agent.tools.zoom import obtain_zoom_in, analyze_zoomed_plots

ALL_TOOLS = [
    plot_light_curve_csv,
    diagnose_plot_with_vlm,
    obtain_GLS,
    obtain_BLS,
    analysis_LS,
    obtain_zoom_in,
    analyze_zoomed_plots,
]

__all__ = [
    "plot_light_curve_csv",
    "diagnose_plot_with_vlm",
    "obtain_GLS",
    "obtain_BLS",
    "analysis_LS",
    "obtain_zoom_in",
    "analyze_zoomed_plots",
    "ALL_TOOLS",
]