---
name: full-light-curve-diagnose
description: Diagnose a variable-star light curve stored as a CSV by plotting it, assessing it with a vision model, and orchestrating follow-up period analysis.
---

# full-light-curve-diagnose

## When to use
Use this skill when the user asks to diagnose a variable source light curve and provides a CSV path (and optionally column names).

## Tools you have
- `plot_light_curve_csv`: Loads CSV and produces a PNG plot.
- `diagnose_plot_with_vlm`: Sends the PNG to a vision model and returns a structured diagnosis text.
- `obtain_GLS`: Computes Generalized Lomb-Scargle periodogram and phase-folded plots.
- `obtain_BLS`: Computes Box-Least-Squares periodogram and phase-folded plots (for transits).
- `obtain_zoom_in`: Produces zoom-in plots for specified time ranges, return the path.
- `analysis_LS`: Analyzes the phase-folded results from GLS or BLS.
- `analyze_zoomed_plots`: Analyzes zoomed-in plots for detailed local features.

## Workflow
1) **Initial Plotting**: 
   Call `plot_light_curve_csv` with:
   - `csv_path`
   - `time_col`, `y_col`, `yerr_col` (if not given, try to read the first row/header of the file).
   - `y_axis_type` = "mag" (if magnitude, invert y-axis) or "flux".
   - `out_path` = `./artifacts/{csv_name}_lc_plot.png`.

2) **Visual Assessment**: 
   Call `diagnose_plot_with_vlm` on the image path returned by step 1. 
   - The output will contain sections on variability type, morphology, and recommendations (GLS/BLS).
3) **CRITICAL DECISION STEP (Do not stop here!)**:
   Read the text output from `diagnose_plot_with_vlm` carefully.
   - **IF** the text says **"GLS Analysis: YES"** (or similar), you **MUST** call `obtain_GLS` immediately.
   - **IF** the text says **"BLS Analysis: YES"** (or similar), you **MUST** call `obtain_BLS` immediately.
   - **IF** the text recommends **Zoom-in visualization** (or similar), you **MUST** call `obtain_zoom_in` immediately. With the `time_ranges` being the list of ranges suggested by the vision model.
   
   *Note: Do not generate the final user response yet. You must run the recommended periodogram tools first.*

4) **Follow-up Analysis (If triggered)**:
   - If `obtain_GLS` was called, take the result and feed it into `analysis_LS` together with 
      - `candidate_type` : a complete GLS candidate list suggested by `diagnose_plot_with_vlm` 
   - If `obtain_BLS` was called, take the result and feed it into `analysis_LS` together with 
      - `candidate_type` : a complete BLS candidate list suggested by `diagnose_plot_with_vlm` 
   - If `obtain_zoom_in` was called, take zoomed-in plot result path together with `candidate_type` as a list of zoom-in candidates suggested by `diagnose_plot_with_vlm` results and feed it into `analyze_zoomed_plots` to get detailed local feature analysis.
      - If the result from `analyze_zoomed_plots` suggests further zoom-in, invoke (3) and (4) again recursively until no further zoom-in is suggested.

5) **Final Reporting**:
   Aggregate all information (Visual Diagnosis + Periodogram Results + Phase Folding Analysis) into the final diagnosis report. Your report must include:
   - Summary diagnosis with confidence level.
   - The most likely classification with reasoning.
   - Key visual evidence (citing the plots generated).
   - The results of the Period analysis (Best Period, Power) if performed.
   - A list of the complete candidate types considered.
   - Level of scientific interest
   - requirements for follow-up observations (if any).