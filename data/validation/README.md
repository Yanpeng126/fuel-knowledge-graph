# Flame-speed comparison data

`flame_speed_figure_data.csv` is the supplied source data for the methane/air and
hydrogen/air flame-speed comparison figures. Numerical values are unchanged.

| Dataset | Rows | Representation |
|---|---:|---|
| CH4_experiment | 98 | Experimental points in FKG |
| CH4_GRI | 19 | Supplied GRI-Mech 3.0 curve samples |
| H2_experiment | 20 | Experimental points in FKG |
| H2_GRI | 25 | Supplied GRI-Mech 3.0 curve samples |

`equivalence_ratio` is dimensionless; `flame_speed_cm_s` is in cm/s. The `doi`
column records source metadata where provided. Reference curves should be plotted
by sorting their equivalence ratios and joining their supplied values, without
extrapolation. This CSV does not encode temperature, pressure or measurement
uncertainties; do not infer those conditions from these columns alone.
