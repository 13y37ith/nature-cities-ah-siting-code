# Nature Cities AH Siting Code

This repository contains the cleaned workflow used to construct accessibility indicators, derive the LQI-based typology, run XGBoost/SHAP analysis and optional GraphSAGE diagnostics, generate abstract hex maps, and export threshold-based screening tables.

## Repository structure

- `01_build_indicators_amap.py`  
  Constructs accessibility and diversity indicators from AMap POI and route services.

- `02_run_lqi_pipeline.py`  
  Computes `lqi_score`, performs KMeans clustering, assigns typology labels, runs XGBoost/SHAP analysis, and optionally runs GraphSAGE.

- `03_plot_hex_maps.py`  
  Generates abstract hex-based maps of Shanghai from the final results table.

- `04_threshold_rule_analysis.py`  
  Produces the median-boundary and incremental rule-screening tables.

## Notes

- API keys are not stored in the scripts. For the AMap workflow, set `AMAP_KEY` as an environment variable or pass it through `--amap-key`.
- Hard-coded personal paths and Colab-specific upload steps have been removed.
- Public-facing type names follow the manuscript:
  - Type A: Resource-Balanced
  - Type B: Transit-Oriented
  - Type C: Culture-Oriented
  - Type D: Resource-Poor
- In the cleaned pipeline, `maturity_score` has been replaced by `lqi_score` to align with the manuscript.
- No genetic algorithm implementation was found in the uploaded scripts. One earlier analysis branch used Optuna for hyperparameter tuning, but no GA-based code is included here.