# This documentation provides the details status

pipeline:

Search for gaia dr3 astrophysical parameters from star name or coordinates.

Calculate Gaia absolute G magnitude from Gaia G and parallax. If a cached Gaia
HR density grid exists, plot the target on the Gaia HR diagram (BP-RP vs M_G)
and send that image to metadata diagnosis with the LAMOST spectrum image.

Build the Gaia HR background grid once with:

```bash
python tools/build_gaia_hr_grid.py \
  --sample-size 1000000 \
  --output-path assets/gaia_hr_diagram/gaia_hr_density_grid.npz \
  --preview-png assets/gaia_hr_diagram/gaia_hr_density_grid.png
```

Search for lamost spectrum data using gaia id

Send the spectrum data, Gaia HR diagram placement, and gaia dr3 astrophysical parameter to model. If not found, also tell model that these data are not found, which is also informative. Give 

{
    "strong_excluded_variable_type":[],
    "reason":str
}


Plot the light curve, send the light curve with the gaia and lamost json results to the model. The model would give the description from the current observartion and suggestions for the next steps.

{
    "require_GLS":bool, 
    "require_GLS_reason":str,
    "require_BLS":bool,
    "require_BLS_reason":str,
    "require_zoom_in":[], # The range of zoom-in if required, in the format of [[t_start, t_end], [t_start, t_end], [t_start, t_end]] if no zoom-in is required, return an empty array.
    "require_zoom_in_reason":str,
    "strong_excluded_variable_type":[str]
}

Reading the json results, the code would use the required tools and generate the result plots. The plots would be sent to the model for further analysis.

For GLS plot diagnosis, the GLS code would return
{
    "result_png_path":str,
    "best_period":float,
    "period_unit":"day",
    "analysis_type":"GLS",
    "selected_period_search_series":str,
    "ignored_periods":{"period_windows_day":[{"period_day":1.0,"tolerance_day":0.01},{"period_day":0.5,"tolerance_day":0.005},{"period_day":0.3333333333333333,"tolerance_day":0.0033333333333333335},{"period_day":2.0,"tolerance_day":0.01},{"period_day":3.0,"tolerance_day":0.01}]},
    "folded_period_metrics":{
        "P": {
            "R21":float,
            "R31":float,
            "phi21_rad":float,
            "phi31_rad":float,
            "amplitude":float,
            "skewness":float,
            "rise_time_phase":float,
            "rise_time_day":float,
            "fall_time_phase":float,
            "fall_time_day":float,
            "phase_of_maximum":float,
            "phase_of_maximum_epoch_day":float,
            "phase_coverage":{"bin_count":int,"occupied_bin_count":int,"fraction":float},
            "scatter_around_folded_model":float,
            "typical_photometric_uncertainty":float|null,
            "scatter_to_uncertainty_ratio":float|null
        },
        "P/2": {},
        "2P": {}
    },
    "metric_definitions":{}
}

the model would accept the image and the previous results and return 
{
    "strong_candidate_type":[],
    "possible_candidate_type":[]
    "strong_excluded_variable_type":[],
    "best_period":"P"/"2P"/"P/2"/"none",
    "Amplitude":float,
    "require_prewhitening":bool,
    "prewhitening_signal_count":int,
    "reason":str
}

If GLS diagnosis requires prewhitening, the pipeline runs a GLS(prewhitening)
follow-up before the final report. GLS(prewhitening) subtracts the requested
number of strongest base periodic Fourier signals. After each base period P is
subtracted, it repeatedly checks the residual GLS periodogram; if the current
most significant residual peak lies within P/n +/- P/n/100 for n=1..10, that
peak is also subtracted. This continues until the current most significant
residual peak is outside all of those P/n windows. It then returns and diagnoses
the residual GLS periodogram plus P, P/2, and 2P folded residual light curves.
Folded GLS and GLS(prewhitening) plots include photometric error bars when
error columns are available.

For BLS diagnosis

the BLS code would return
{
    "result_png_path":str,
    "best_period":float,
    "period_unit":"day",
    "optimal_depth":float,
    "optimal_duration":float,
    "transit_epoch":float,
    "transit_start_epoch":float,
    "best_power":float,
    "search_device":str,
    "ignored_periods":{"period_windows_day":[{"period_day":1.0,"tolerance_day":0.01},{"period_day":0.5,"tolerance_day":0.005},{"period_day":0.3333333333333333,"tolerance_day":0.0033333333333333335},{"period_day":2.0,"tolerance_day":0.01},{"period_day":3.0,"tolerance_day":0.01}]}
}

the model would accept the image, the previous results, and the remaining
candidate pool and return
{
    "strong_excluded_variable_type":[],
    "period":"P"/"2P"/"P/2"/"none",
    "reason":str
}

For zoom-in diagnosis
{
    "strong_candidate_type":[],
    "possible_candidate_type":[],
    "strong_excluded_variable_type":[],
    "flux_changing_continuously":bool,
    "need_further_zoom_in":[[range for further zoom in]],
    "reason":str
}


After all the analysis is complete, the model would give the final diagnosis of the variable type of the star. The result would be in the format of 
{
    "final_candidate_type":str,
    "Need_more_observation":bool,
    "follow_up_observation_type":{"optical in band X/ spectroscopy with X resolution":"urgent"/"normal"},
    "scientific_importance":"high"/"medium"/"low",
    "reason":str
}
