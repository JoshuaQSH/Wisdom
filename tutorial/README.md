# Cross-layer WISDOM testing on tabular data

This tutorial trains the quantile DNN in `dnn.py` on two datasets independently. The education and healthcare cases predict `stress_index` from `minHr`, `maxHr`, `avgHr`, and `medianHr`. The midpoint of the DNN's two quantile outputs is the point prediction.

## Run both use cases

Place the supplied `ai4work_education_completed.csv` and `ai4work_healthcare_completed.csv` in `tutorial/healthcare_datasets/`. From the Wisdom repository root, run both commands with `uv`.

```sh
uv run --locked python tutorial/run_wisdom_healthcare_pretrain.py --dataset all
uv run --locked python tutorial/run_wisdom_healthcare_testing.py --dataset all
```

With the [project environment](../README.md#prerequisites) activated, run the same script commands starting with `python tutorial/...`. Pass `--dataset education` or `--dataset healthcare` to run one case. Pretraining defaults to 200 epochs, seed 42, and the 5th and 95th quantile heads.

## Test with saved artifacts

```sh
uv run --locked python tutorial/run_wisdom_healthcare_testing.py \
  --weights-path tutorial/weights/education_dnn.pt \
  --test-data-path tutorial/healthcare_datasets/ai4work_education_completed.csv \
  --wisdom-csv tutorial/importances/education_wisdom.csv
```

When the CSV matches the original pretraining input, testing reproduces the participant holdout. You can also pass an already-held-out CSV containing the four feature columns and `stress_index`. Testing applies the saved training preprocessing and cluster centers directly to those rows. In that CSV, `userId` is optional and supplies a participant count for the report.

## How it works

1. **Prepare the data.** `utils.py` drops rows without a usable target or heart-rate measurement and holds out roughly 20% of participants for testing. It fits median imputation and standardization on the training split and applies those transforms to the held-out rows.
2. **Train and score.** Pretraining fits the DNN with pinball loss. WISDOM combines Layer DeepLift, Layer Integrated Gradients, and Layer Gradient × Activation on the training rows. It saves the score CSV and a plot of the ten highest scores in `importances/`. It fits two KMeans clusters per neuron for the top three positively scored neurons per hidden layer and saves the centers with the model checkpoint in `weights/`.
3. **Measure cross-layer coverage.** Each test row contributes one joint tuple of cluster assignments across the selected layers. WISDOM coverage is the fraction of possible tuples observed on the test set. The maximum for the suite is bounded by the number of test rows.

The testing command prints a summary for each case and writes three files under `results/`.

| File | Contents |
| --- | --- |
| `<dataset>_coverage.csv` | Overall coverage, selection counts, Acc@5, and Acc@10 |
| `<dataset>_coverage.json` | Coverage, selected neuron indices, inputs, and prediction metrics |
| `<dataset>_suite_metrics.csv` | Coverage and prediction metrics for the test suite |

Acc@5 and Acc@10 are the fractions of test predictions within 5 and 10 stress-index units of the supplied target, respectively, including the boundary. The completed datasets include imputed stress scores.

## Results

The default run (200 epochs, seed 42) selected nine neurons across three hidden layers for each dataset.

| Dataset | Test rows | Combinations observed | Cross-layer WISDOM coverage | Acc@5 | Acc@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Education](results/education_coverage.json) | 919 | 27 / 512 | 5.27% | 27.97% | 54.30% |
| [Healthcare](results/healthcare_coverage.json) | 771 | 31 / 512 | 6.05% | 43.45% | 81.84% |
