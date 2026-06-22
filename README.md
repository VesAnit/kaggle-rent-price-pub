# Moscow apartment price prediction

Kaggle-style regression task: estimate the sale price of a Moscow apartment from a property listing and macroeconomic context.

**Notebook:** `blending_cb_eng.ipynb` (English) · `blending_cb.ipynb` (Russian)

---

## The competition

The goal is to predict how much an apartment actually sold for.

Each row is a listing with physical characteristics (area, rooms, floor, building year, condition, materials), location (`sub_area` — a Moscow district), and the date of the transaction. Macroeconomic indicators (rent levels, wages, consumer basket, life expectancy) are available by date and can be linked to listings.

The model receives `train.csv` with known prices and must fill in prices for `test.csv`. The submission format is two columns: `id` and `price_doc`.

**Target — `price_doc`:** the documented sale price of the apartment in rubles. This is the real transaction price from the listing, not rent and not a price per square metre.

**Evaluation metric:** MAE (mean absolute error between predicted and actual price).

---

## Approach

### Exploratory analysis

Distributions, outliers, missing values, and inconsistent records were reviewed first (e.g. living area plus kitchen larger than total area, floor above building height). Findings from EDA informed the cleaning rules below.

### Data cleaning and imputation

A preprocessing pipeline:

- fixes implausible values and logical contradictions in area and floor fields;
- fills gaps using district-level statistics first (median/mode by `sub_area`), then global fallbacks;
- reconstructs missing areas from related fields where possible (e.g. total area from living + kitchen area).

Local market context is used throughout — apartments in the same district are treated as more comparable than city-wide averages.

### Feature engineering

Three groups of signals:

1. **From the listing itself** — ratios and interactions: building age, area per room, floor position, kitchen/living share of total area, area bins, and similar derived fields.
2. **From macro data** — economic environment at the time of sale: rent benchmarks, salary, basket, life expectancy.
3. **From the district** — how expensive and how active the area is: target encoding of `sub_area` with smoothing for rare districts, median/mean price aggregates, listing count, district popularity.

Macro features are joined by transaction date. Categorical fields (`material`, `state`, area bin) are kept for tree models that handle categories natively.

### Modelling

Gradient boosting regressors:

- two **CatBoost** models with different hyperparameters (one tuned with Optuna, one set manually for diversity);
- one **LightGBM** model.

All optimize MAE. Predictions from the three models are combined in a **weighted blend**; weights are chosen on a hold-out validation split. K-fold cross-validation is available to check stability across folds.

### Code structure

The notebook implements a single end-to-end pipeline (`PricePredictionPipeline`): preprocessing → feature engineering → ensemble training → submission export.

---

## Data

```
mlurfuflat/
├── train.csv      # listings with price_doc
├── test.csv       # listings without price
├── macro.csv      # macro indicators by date
└── submission.csv # submission template (dummy values)
```

Model output is written to `submission.csv` in the project root.
