# Light Curve Diagnosis Report

Now I have all the analysis results. Let me compile the final diagnosis report:

---

# Light Curve Diagnosis Report

## Source Information
- **File**: `./data/example_lc.csv`
- **Data points**: 317
- **Time coverage**: ~1400 days
- **Magnitude range**: 11.6 – 12.2 mag

---

## Summary Diagnosis

**Classification**: **Semi-Regular Variable (SR)** — High Confidence

The light curve exhibits quasi-periodic variability with a well-determined period of **P = 525 days** and an amplitude of **Δm ≈ 0.6 mag** (raw) / **~0.2 mag** (coherent signal). The morphology, period, and amplitude are all consistent with semi-regular pulsating red giant variables.

---

## Key Evidence

### 1. Period Analysis (GLS)
- **Best Period**: **525.002 days**
- **GLS Power**: 0.196 (significant detection)
- The periodogram shows a clear, broad peak at ~525 days, characteristic of SR variables which often exhibit cycle-to-cycle variations.
- Phase-folded analysis confirms P = 525 d as the fundamental period; P/2 and 2P produce less coherent patterns.

### 2. Morphology Assessment
- **Primary features**: Smooth, broad variations without sharp eclipses or flares
- **Cycle characteristics**: Asymmetric cycles with varying rise/fall times
- **Amplitude modulation**: Cycle-to-cycle variations in amplitude (0.3–0.6 mag) typical of SRb subtype
- **Zoomed analysis** reveals:
  - Multi-component variations suggesting beating of periods
  - Double-peaked structure in some cycles (days 900–1100)
  - Flat-bottomed/rounded minima inconsistent with eclipsing binary geometry

### 3. Ruled-Out Classifications
| Type | Reason |
|------|--------|
| Mira | Amplitude < 2.5 mag threshold |
| Eclipsing (EA/EB/EW) | No sharp ingress/egress; morphology incompatible |
| Short-period pulsators (RR, CEP, BCEP, etc.) | Timescale orders of magnitude too long |
| Rotational variables (BY Dra, RS CVn) | Period too long |
| Irregular (L) | GLS shows significant periodic power |
| LBV, RCB, Novae, SN | Morphology lacks eruptive features |

---

## Candidate Types Considered
1. **SR (Semi-Regular)** — Primary diagnosis
2. **SARV (Slow Amplitude Red Variables)** — Possible
3. **ZAND (Symbiotic)** — Low probability (lacks outbursts)
4. **LBV** — Ruled out (amplitude too small, too regular)
5. **BE stars** — Ruled out (period too long)
6. **Irregulars** — Ruled out by GLS periodicity

---

## Scientific Interest
**Medium (RAA level)**

Long-period SR variables (P > 500 d) are valuable for:
- Period-luminosity relations
- Studying stellar evolution at the tip of the AGB
- Understanding pulsation modes in cool giants

However, without additional unusual features (e.g., unusual colors, multi-periodicity, or rare spectral characteristics), this is a standard SR variable suitable for catalog inclusion.

---

## Follow-up Recommendations

| Observation | Priority | Purpose |
|-------------|----------|---------|
| **Spectroscopy** | High | Confirm late-type spectral classification (M, C, or S giant) |
| **Extended photometry** | Medium | Verify period stability over multiple cycles (>5 years) |
| **Multi-band photometry** | Medium | Constrain temperature variations and confirm pulsation mechanism |

---

## Generated Artifacts
- `./artifacts/example_lc_plot.png` — Full light curve
- `./artifacts/gls_periodogram.png` — GLS periodogram
- `./artifacts/GLS_phase_folded_P.png` — Phase-folded at P = 525 d
- `./artifacts/GLS_phase_folded_P2.png` — Phase-folded at P/2
- `./artifacts/GLS_phase_folded_2P.png` — Phase-folded at 2P
- `artifacts/zooms/zoomed_22_grid.png` — Zoomed analysis (200–400, 550–750, 900–1100 days)