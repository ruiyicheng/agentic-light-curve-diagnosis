---
name: full-light-curve-diagnose
description: Diagnose a variable-star light curve stored as a CSV by plotting it, assessing it with a vision model, and orchestrating follow-up period analysis.
---

# full-light-curve-diagnose

## When to use
Use this skill when the user asks to diagnose a variable source light curve and provides a CSV path (and optionally column names).

## Tools you have
- `inspect_light_curve_schema`: Read the file head, infer delimiter/header/columns, detect source metadata, and report whether the data is multi-band.
- `lookup_source_context`: Resolve source name/coordinates, search VSX first, and fetch Gaia astrometry if possible.
- `plot_light_curve_csv`: Loads CSV and produces a PNG plot.
- `diagnose_plot_with_vlm`: Sends the PNG to a vision model and returns a structured diagnosis text.
- `obtain_GLS`: Computes Generalized Lomb-Scargle periodogram and phase-folded plots.
- `obtain_BLS`: Computes Box-Least-Squares periodogram and phase-folded plots (for transits).
- `obtain_zoom_in`: Produces zoom-in plots for specified time ranges, return the path.
- `analysis_LS`: Analyzes the phase-folded results from GLS or BLS.
- `analyze_zoomed_plots`: Analyzes zoomed-in plots for detailed local features.

## Workflow
1) **Inspect the file first**:
   Call `inspect_light_curve_schema` with the CSV path before any other tool.
   - Use the detected `time_col`, `y_col`, `yerr_col`, `band_col`, `scale`, and any source metadata it finds.
   - If a filter/band column is present, treat the file as multi-band. Do not drop the band information.

2) **Search catalogs before diagnosis**:
   If the file head, user request, or schema inspection provides a source name or coordinates, call `lookup_source_context` immediately.
   - If VSX or Gaia information is available, you must use it in the reasoning and final report. Do not ignore catalog evidence.
   - VSX is the gate. If VSX returns a known variable match, skip further diagnosis work such as plotting for classification, GLS, BLS, or zoom recursion.
   - In that case, produce a report centered on the catalog identification, Gaia context (if returned), and any quick sanity notes from the light-curve schema.
   - Gaia context should be used when available to report parallax/proper motion plus diagnosis-relevant astrophysical hints such as `Teff`, `logg`, likely galaxy vs star, and likely binary/non-single-star status.
   - If VSX does not identify a known variable, keep the Gaia context in the final report and continue.

3) **Initial Plotting**:
   Call `plot_light_curve_csv` with the detected columns from step 1. Let the tool keep multi-band data together if a band/filter column exists.
   - Always pass a unique `out_path` for the diagnosis plot.
   - The output path must not be a shared generic filename. Use a path derived from the input file stem or source identifier, for example `./artifacts/{csv_stem}_diagnose_plot.png`.

4) **Visual Assessment**:
   Call `diagnose_plot_with_vlm` on the image path returned by step 3.
   - The output will contain sections on variability type, morphology, and recommendations (GLS/BLS).

5) **CRITICAL DECISION STEP (Do not stop here!)**:
   Read the text output from `diagnose_plot_with_vlm` carefully.
   - **IF** the text says **"GLS Analysis: YES"** (or similar), you **MUST** call `obtain_GLS` immediately.
   - **IF** the text says **"BLS Analysis: YES"** (or similar), you **MUST** call `obtain_BLS` immediately.
   - **IF** the text recommends **Zoom-in visualization** (or similar), you **MUST** call `obtain_zoom_in` immediately. With the `time_ranges` being the list of ranges suggested by the vision model.
   
   *Note: Do not generate the final user response yet. You must run the recommended periodogram tools first.*

6) **Follow-up Analysis (If triggered)**:
   - If `obtain_GLS` was called, take the result and feed it into `analysis_LS` together with 
      - `candidate_type` : a complete GLS candidate list suggested by `diagnose_plot_with_vlm` 
      - Trust the GLS result when the binned points show a clear, coherent trend in any one of the `P`, `2P`, or `P/2` phase-folded panels.
      - GLS has higher priority than zoom-in morphology for periodic classification. If GLS shows a convincing folded trend, prefer the GLS-supported interpretation even if zoom-in morphology is ambiguous.
   - If `obtain_BLS` was called, take the result and feed it into `analysis_LS` together with 
      - `candidate_type` : a complete BLS candidate list suggested by `diagnose_plot_with_vlm` 
   - If `obtain_zoom_in` was called, take zoomed-in plot result path together with `candidate_type` as a list of zoom-in candidates suggested by `diagnose_plot_with_vlm` results and feed it into `analyze_zoomed_plots` to get detailed local feature analysis.
      - Zoom-in analysis is secondary to GLS for periodic variables.
      - Because the real period can be shorter than the zoom-in range, zoomed plots may fail to show the full repeating pattern even when GLS has identified the correct period.
      - Use zoom-in analysis mainly to inspect local morphology, data quality, and whether a proposed periodic interpretation is visibly contradicted by obvious local structure.
      - Be conservative: if no significant trend or coherent local morphology is observed after zoom-in, do not overrule a convincing GLS result and do not escalate the classification strength based on zoomed plots alone.
      - If the result from `analyze_zoomed_plots` suggests further zoom-in, invoke (3) and (4) again recursively until no further zoom-in is suggested.

7) **Final Reporting**:
   Aggregate all information (schema inspection + VSX/Gaia context + visual diagnosis + periodogram results + phase folding analysis) into the final diagnosis report. Your report must include:
   - The columns and scale chosen from the file head, including whether the light curve is multi-band.
   - The path to the diagnosis plot if one was generated.
   - VSX result and whether further diagnosis was skipped.
   - If VSX is available, explicitly use the VSX class, period, epoch, and match status in the diagnosis reasoning.
   - Write all returned VSX query results into the final report when they are available. Do not omit returned online-query fields; include the full VSX match payload in a dedicated report section.
   - A clear source-status statement:
     - If VSX identifies the source, explicitly state that it is an already cataloged / confirmed known variable rather than a new discovery.
     - If VSX does not identify the source, explicitly state that it remains an uncataloged candidate and that the classification is a new diagnosis candidate rather than a confirmed catalog identification.
   - Gaia astrometric and astrophysical parameters if they were available, including `Teff`, `logg`, and galaxy/binary hints when Gaia provides them.
   - If Gaia astrophysical parameters are available, explicitly use them in the diagnosis reasoning to support or weaken candidate classes. This includes at minimum stellar/non-stellar interpretation, galaxy likelihood, binary likelihood, and any useful `Teff`/`logg` constraints.
   - Write all returned Gaia query results into the final report when they are available. Do not omit returned online-query fields; include the full Gaia payload in a dedicated report section, and separately summarize the most diagnosis-relevant entries in the narrative.
   - Summary diagnosis with confidence level.
   - The most likely classification with reasoning.
   - If GLS and zoom-in evidence disagree, explain that GLS has higher priority for periodic diagnosis unless zoom-in reveals a clear contradiction or data-quality failure.
   - Key visual evidence (citing the plots generated).
   - Embed the diagnosis plot in the markdown report when the plot path is available, using standard markdown image syntax with the generated artifact path.
   - The results of the period analysis (best period, power, backend) if performed.
   - A list of the complete candidate types considered.
   - Level of scientific interest
   - requirements for follow-up observations (if any), and how urgent is the follow-up observation with reason.
