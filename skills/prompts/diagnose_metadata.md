## Input

The input is the full JSON object returned by `lookup_source_catalogs`, usually containing:

- `resolved_source`: coordinate resolution details.
- `vsx`: known-variable matches and `is_known_variable`.
- `gaia`: Gaia DR3 astrometry, photometry, astrophysical parameters, classification probabilities, `absolute_g_mag`, HR diagram coordinates with formal uncertainties when available, and `diagnosis_hints`; may be `null` or an error object.
- `gaia_hr_diagram`: cached Gaia HR density-grid plot status and target BP-RP/M_G coordinates, including an uncertainty error bar when available; may be unavailable if the grid has not been built.
- `lamost`: LAMOST lookup status, match metadata, and spectrum CSV/PNG paths; may be unavailable or an error object.
- `skip_further_diagnosis`: true when VSX already identifies the object as a known variable.

The given images are the LAMOST spectrum plot and Gaia HR diagram if available.

Complete possible hypotheses type should be listed within:

<candidate_pool>

## Astrophysical Consistency Rules

Use metadata as a physical-constraint filter before any morphology-based
classification step. The purpose of this step is not to pick a final positive
class, but to remove classes that are astrophysically incompatible with the
source.

Evaluate the following parameters when available:

- Stellar temperature or spectral type: hot-star classes require hot spectra;
  cool-star activity classes require late-type stars; Delta Scuti/SX Phoenicis
  and related short-period pulsators require A/F-type or otherwise compatible
  blue stars.
- Surface gravity, radius, luminosity class, and absolute magnitude:
  high-luminosity pulsators and evolved variables require giant or supergiant
  parameters, while dwarf activity and many close-binary interpretations require
  main-sequence or compact-system parameters.
- Gaia color, parallax, distance, absolute magnitude, and Gaia HR diagram
  position: use these together to distinguish dwarfs, giants, white dwarfs,
  hot OB stars, and luminous evolved stars. If HR placement uncertainties are
  provided (`bp_rp_error`, `absolute_g_mag_error`, or an error bar on the HR
  diagram image), use the uncertainty range when deciding whether a class is
  physically incompatible. Do not rely on apparent magnitude alone.
- LAMOST spectrum, when available: use spectral type, emission lines,
  Balmer/He features, molecular bands, and gravity-sensitive features to reject
  incompatible classes.
- Catalog classifications and probabilities: use them only when they are
  consistent with the physical parameters, not as unverified labels.

Strongly exclude a variable type only when the catalog or spectrum makes the
stellar parameters decisively incompatible with that type. If the metadata are
ambiguous, uncertain, missing key parameters, or only weakly inconsistent with
the class, keep the class possible. Examples: a cool high-gravity M dwarf rules
out Cepheids, RR Lyrae, Delta Scuti/HADS, hot OB pulsators, Mira/SR giant
variables, and luminous blue variables; a hot luminous OB star rules out cool
dwarf spot-modulation classes; a white-dwarf spectrum supports
compact-pulsator or compact-binary constraints and rules out normal giant or
main-sequence pulsator classes.

When the metadata are incomplete or uncertain, keep the candidate rather than
excluding it. State clearly which parameters were used in `reason`, especially
Teff, spectral type, logg, absolute magnitude, color, and spectrum availability.

## Output
Return only valid JSON with this exact schema:

```json
{
  "strong_excluded_variable_type": [],
  "reason": ""
}
```

Do not add markdown, comments, prose outside the JSON, or extra keys. Metadata
and spectra may strongly rule out incompatible classes, but they must not be
used to declare a strong positive variable-type candidate. Use an empty
`strong_excluded_variable_type` array when the catalog evidence is not strong
enough for exclusion.


## Availability Checks

Treat Gaia as found only when `gaia` is an object without `error` and has a non-empty `source_id`.

Treat LAMOST spectrum data as found only when `lamost.available` is true and at least one of `lamost.spectrum_csv_path`, `lamost.spectrum_csv_relative_path`, `lamost.spectrum_png_path`, or `lamost.spectrum_png_relative_path` is present. If `lamost.match` exists but `available` is false, use it only as weak partial metadata and state that no spectrum was available.

Treat Gaia HR diagram placement as found only when `gaia_hr_diagram.available` is true and `gaia_hr_diagram.target.absolute_g_mag` and `gaia_hr_diagram.target.bp_rp` are present. Use this placement and its uncertainties, when present, as supporting physical context for luminosity class and temperature, not as a catalog label. If the target error bar overlaps multiple evolutionary regions, avoid strong exclusions that depend on a precise HR position.

The results from lookup_source_catalogs is as follows:
