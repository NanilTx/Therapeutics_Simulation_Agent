# Pipeline Results

## Selected Candidate

| target | dir | dose | duration | score |
| ------ | --- | ---- | -------- | ----- |
| MAPT   | ↓   | 1.0  | 14       | 1.899 |

Validation RMSE: `0.189` (lower is better)

## Biomarker Effects (first 3 of 3)

| index | delta  | uncertainty |
| ----- | ------ | ----------- |
| 1     | -0.089 | 0.059       |
| 2     | 0.003  | 0.050       |
| 3     | 0.041  | 0.054       |

Score breakdown: `effect_sum=0.352`, `uncertainty_sum=0.185`, `ratio=1.899`

## Top 5 Candidates

| rank | target | dir | dose | duration | score |
| ---- | ------ | --- | ---- | -------- | ----- |
| *1   | MAPT   | ↓   | 1.0  | 14       | 1.899 |
|  2   | APP    | ↓   | 1.0  | 7        | 1.512 |
|  3   | MAPT   | ↓   | 0.5  | 7        | 0.950 |
|  4   | MAPT   | ↓   | 0.25 | 28       | 0.817 |
|  5   | APP    | ↓   | 0.25 | 14       | 0.654 |

Gap to next best: `Δ=0.387` (`25.58%`)

## Summary

Evaluated 6 candidate interventions and selected target MAPT (downregulation) at dose 1.0 for 14 days. Selection score: 1.899 (higher is better). Predictions include 3 biomarker effect values with uncertainties. Retrospective validation RMSE: 0.189.

## Narrative

We proposed 6 interventions and simulated expected biomarker effects with uncertainties. We prefer candidates with strong overall effect and lower total uncertainty. Based on this, we selected MAPT at dose 1.0 for 14 days. A retrospective check suggests the model’s predictions are reasonably calibrated (see RMSE).

_Legend: dir ↑ upregulation, ↓ downregulation. Score = effect_sum / uncertainty_sum (higher is better)._

## Configuration

| param      | value                                                      |
| ---------- | ---------------------------------------------------------- |
| n          | 6                                                          |
| seed       | 1337                                                       |
| latent_dim | 16                                                         |
| data_dir   | /Users/biniraj/Downloads/therapeutic-simulation-agent/data |
| elapsed_s  | 0.006                                                      |
