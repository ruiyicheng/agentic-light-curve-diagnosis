Summarize the whole previous light curve analysis results, including metadata diagnosis, full-plot diagnosis, GLS diagnosis, and BLS diagnosis. Provide a concise summary of the most likely variable type(s) and the reasoning behind it. Also, suggest any final follow-up steps or observations needed to confirm the classification.

The summary should be in the following JSON format:

```json
{
  "most_likely_variable_type": str,
  "confidence_level": "low"/"medium"/"high",
  "other_possible_variable_types": [str],
  "Scientific interest":"low"/"medium"/"high",
  "requirement for future observation": [str],
  "urgency_level_for_follow_up": ["urgent"/"normal"/"low"],
  "reasoning": str
}
```

Rules:
- Use the provided `Candidate pool from previous steps` as the active candidate set at final-report time.
- The final diagnosis output must stay within `ALL_VARIABLE_TYPES`. In practice, choose labels from the provided candidate pool (which is derived from `ALL_VARIABLE_TYPES`).
- `most_likely_variable_type` must be one single strong candidate label. Do not join labels with `|`, do not output hybrid labels, and do not list weak speculative alternatives as final classifications.
- The labels in `most_likely_variable_type` and `other_possible_variable_types` should come from that candidate pool unless there is explicit evidence in the diagnosis JSON that strongly supports an out-of-pool interpretation.
- `confidence_level` should reflect how strongly the evidence supports the most likely variable type.
- `other_possible_variable_types` should be empty unless another type remains a strong, astrophysically compatible candidate after applying all constraints. Do not include types that are merely possible from morphology alone.
- Be conservative with final strong conclusions. Use `high` confidence only when the period, morphology, metadata, and any prewhitening results agree decisively. If important evidence is noisy, alias-like, broad compared with uncertainties, or dependent on marginal exclusions, use lower confidence and preserve plausible alternatives in the reasoning.

Astrophysical decision rules:
- A final candidate is strong only when both conditions are satisfied: the light-curve evidence supports the class, and the stellar astrophysical parameters do not contradict that class.
- Use Gaia and LAMOST metadata as hard physical constraints when available: Teff, spectral type, logg, radius, luminosity class, absolute magnitude, color, parallax/distance, spectrum features, and emission lines.
- When light-curve morphology cannot distinguish between candidate classes, choose the class whose required stellar parameters best match the metadata. For example, use stellar temperature, gravity, color, and luminosity to break degeneracies such as EW versus HADS/DSCT, RR Lyrae versus contact binary, hot-star pulsator versus cool-dwarf activity, and giant pulsator versus dwarf binary.
- Do not promote a morphology-compatible class to final candidate if its required stellar type, evolutionary stage, luminosity, or spectrum is inconsistent with the metadata.
- If no non-constant variable type is strongly supported after applying morphology and astrophysical constraints, choose `CST` only when the candidate pool permits it and explain that the available evidence is insufficient for a strong variable-star classification.

- `Scientific interest` should reflect whether this target align with the given scientific goals and priorities.
- `requirement for future observation` should list any necessary follow-up observations or analyses needed to confirm the classification.
- `urgency_level_for_follow_up` should indicate how quickly the follow-up observations should be conducted based on the type of the target and the confidence level of the classification. For example, a early supernova candidate might require urgent follow-up, while a likely non-variable star with low scientific interest might not require any follow-up.
- `reasoning` should summarize the key evidence from the metadata, full-plot, GLS, and BLS analyses that led to the classification and decision. It must explicitly mention the decisive astrophysical parameters used to accept or reject the final candidate when such parameters are available.

When the proposed type is not transiting exoplanet, the BLS results should not be treated as strong evidence. When it contradicts the GLS in period, the results of BLS should be ignored because the GLS provides more reliable period information in non-exoplanetary systems. The final classification should be based on the overall evidence from all analyses, with appropriate weighting of the different types of evidence.
