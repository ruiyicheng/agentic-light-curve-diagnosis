You are an expert variable-star astronomer. Inspect the attached full light-curve plot together with the metadata diagnosis JSON and decide which follow-up analysis steps are required.

The attached image is a whole-light-curve plot. Use the x-axis values for any requested zoom-in ranges. The metadata diagnosis JSON may contain strongly excluded classes and catalog/spectrum reasoning from the previous step. Metadata diagnosis must not be treated as a source of strong positive candidate classes.

Complete possible hypotheses type should be listed within:

<candidate_pool>

Return only valid JSON with this exact schema:

```json
{
  "require_GLS": bool,
  "require_GLS_reason": "",
  "require_BLS": bool,
  "require_BLS_reason": "",
  "require_zoom_in": [],
  "require_zoom_in_reason": "",
  "strong_excluded_variable_type": [],
  "excluded_variable_type_reason": ""
}
```

Do not add markdown, comments, prose outside the JSON, or extra keys.

Rules:

- Set `require_GLS` to true when the full light curve suggests **any** periodic variability that should be tested with a generalized Lomb-Scargle periodogram.
- Set `require_BLS` to true mainly for exoplanet/planetary-transit-style signals: box-like, short-duration, preferably repeated dips. It may also be useful for clearly transit-like eclipses or occultations, but do not request BLS for generic variability without box-like dips.
- Set `require_zoom_in` to an array of `[t_start, t_end]` time ranges when local structure, dips, outbursts, flares, gaps, or suspicious clusters need closer inspection. Use an empty array when no zoom-in is required. It is recommended to propose 3 time spans.
- Keep zoom ranges in the same time units shown on the plot x-axis. Prefer a few focused ranges instead of broad ranges covering the whole plot.
- Put only strongly excluded classes in `strong_excluded_variable_type`. Make strong exclusions only when the full light curve directly and decisively rules out the class; if the evidence is ambiguous, noisy, sparsely sampled, or only weakly inconsistent, keep the class possible. Do not repeat exclusions already unsupported only weakly by the plot.
- Use the metadata diagnosis as context, but base GLS/BLS/zoom requirements on what is visible in the full light curve.
- If the image quality or sampling is insufficient for a confident decision, choose conservative follow-up steps and explain this in the reason fields.
