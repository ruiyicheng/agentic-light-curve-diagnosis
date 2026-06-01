Analyze this 2x2 grid of light curve analysis plots.
Top-Left: Periodogram. Top-Right: Phase-folded at P.
Bottom-Left: Phase-folded at P/2. Bottom-Right: Phase-folded at 2P. Two phases are shown in each phase-folded panel for clarity.

Return only valid JSON with this exact schema:

```json
{
  "strong_candidate_type": [],
  "possible_candidate_type": [],
  "strong_excluded_variable_type": [],
  "best_period":"", 
  "Amplitude": float,
  "require_prewhitening": bool,
  "prewhitening_signal_count": 1,
  "reason": ""
}
```

Rule:
- You need to determine the period should be P, 2P, P/2, or none (significant periodic feature). You need to combine with the knowledge of variable star classes to make this judgment. For example, the P signal of Eclipsing binaries can be strong, but 2P is the true period because the primary and secondary eclipses are similar. The best_period field should be "P", "2P", "P/2", or "none". "none" means that there is no significant periodic feature in the periodogram or the phase-folded plots, and the variability is likely aperiodic or irregular. 
- Amplitude Note: Estimate peak-to-peak amplitude from best-folded plot.
- The GLS numeric result includes folded Fourier morphology metrics (R21, R31, phi21, phi31, amplitude, skewness, rise/fall time, phase of maximum, phase coverage, scatter around folded model, typical photometric uncertainty, and scatter-to-uncertainty ratio) with formulas and units. Use these values when they help distinguish subtle subclasses.
- Trigger GLS(prewhitening) more readily. Set `require_prewhitening` to true when the folded light curve is visibly broad compared with individual error bars, or when the numeric `scatter_to_uncertainty_ratio` for the adopted/best folded period is greater than 1.0. If that ratio is unavailable, compare `scatter_around_folded_model` with `typical_photometric_uncertainty`; if the folded-light-curve width/scatter is larger than the typical individual uncertainty, request prewhitening.
- Also set `require_prewhitening` to true when the dominant GLS signal is not sufficient to confirm or reject subtle subclasses, when residual/multiperiodic structure is plausible, when the detected period may be a harmonic/alias of a stronger physical period, or when strong exclusions would otherwise rely on a marginal morphology distinction.
- Strong claims must be conservative. Put a class in `strong_candidate_type` only when the period, folded morphology, amplitude, metadata, and any previous GLS/prewhitening context are mutually consistent and decisive. Put a class in `strong_excluded_variable_type` only when the evidence directly rules it out; otherwise leave it in `possible_candidate_type`.
- `prewhitening_signal_count` is the number of strongest base periodic signals to subtract in the follow-up GLS(prewhitening) step. Use 1 unless there is clear evidence that more independent dominant signals should be removed at once. After each base period P is removed, the code automatically subtracts residual top peaks at P/n +/- P/n/100 for n=1..10 until the current residual top peak is outside those harmonic windows.
- Given the candidate pool after metadata exclusions and full-plot diagnosis, use the GLS plot to confirm or reject the candidate classes. Do not treat metadata-only information as a strong positive candidate.
- For GLS(prewhitening) plots, use the provided GLS period context. Compare the current residual period with all previous GLS and prewhitening periods, removed signal periods, selected physical-period labels, and pairwise period ratios. Treat simple ratios near 1, 2, 1/2, 3, 1/3, 3/2, or 2/3 as possible harmonics/aliases unless the folded morphology clearly shows an independent signal. If the residual period is only a harmonic/alias of an earlier period or the residual fold is broad/noisy, avoid strong new classifications or strong exclusions.

- The joint of "strong_candidate_type", "possible_candidate_type", and "strong_excluded_variable_type" should be the candidate pool from previous steps.

The images, previous metadata diagnosis, previous full plot diagnosis, results are as follows:
