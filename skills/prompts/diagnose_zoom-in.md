Analyze this 2x2 light curve grid.
Top-Left: Full survey. Top-Right, Bottom-Left, Bottom-Right: Zoom-ins.



Rules:
1. **Flux changing continuously**: Decide this conservatively from the zoom-in
   panels, not from the full-survey panel. Set `flux_changing_continuously` to
   true only when the candidate-relevant feature is visibly resolved by several
   neighboring points that trace a smooth local path in flux.

   Set `flux_changing_continuously` to false if any of these apply:
   - the feature is defined by one, two, or only a few isolated points;
   - neighboring points jump up/down by more than their measurement uncertainty;
   - the points look jagged, alternating, stochastic, or dominated by scatter;
   - there are large time gaps across the feature;
   - the apparent feature could be an outlier, instrumental jump, or
     undersampled event;
   - error bars overlap the proposed morphology enough that ingress, egress,
     decay, or curvature is not clearly resolved.

   When uncertain, choose false. A broad trend, a cluster of points, or a
   plausible astrophysical interpretation is not enough; the plotted zoom-in
   points must directly show a smooth continuous flux change. If false, return
   no strong candidates, no strong exclusions, and no further zoom-in ranges.

2. **Local Morphology** (if flux is changing continuously): Describe shapes in zooms
   (ingress/egress symmetry, flare decay, occultation, stochastic noise).

3. **Consistency** (if flux is changing continuously): Are zoom features representative
   of the whole or unique events?

4. **Data Integrity**: Local outliers or instrumental jumps.

5. **Conclusion** (if flux is changing continuously): How do details refine classification?
   Strong candidates and strong exclusions must be conservative: use
   `strong_candidate_type` only when the zoom morphology directly and
   decisively supports the class, and use `strong_excluded_variable_type` only
   when the zoom morphology directly rules out the class. Otherwise keep the
   class in `possible_candidate_type`.

6. **Further zoom-in** (if flux is changing continuously): Determine if smaller time ranges
   need inspection. If so, specify 3 time ranges [t_start, t_end] in days.
   Only used if data points are still too crowded to inspect.

The output of the analysis should be a JSON with this exact schema:

```json
{
    "strong_candidate_type":[],
    "possible_candidate_type":[],
    "strong_excluded_variable_type":[],
    "flux_changing_continuously": bool,
    "need_further_zoom_in":[[range for further zoom in]],
    "reason":str
}
```
The candidates, reason for candidates, and the zoom-in images are as follows:
