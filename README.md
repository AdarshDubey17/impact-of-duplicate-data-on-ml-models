# Impact of Duplicate Training Samples on Machine Learning Generalization

An experimental study investigating how adding duplicate training records affects the performance and generalization of different machine learning classifiers.

## Problem statement

To study how increasing the proportion of duplicate records in the training data affects the predictive performance and generalization of different machine learning models.

## Objectives

- Compare Logistic Regression, Decision Tree, Random Forest and K-Nearest Neighbors (KNN).
- Test duplicate levels of 0%, 10%, 25%, 50%, 75% and 100%.
- Evaluate model performance using multiple metrics.
- Examine training time and the trainingâ€“testing generalization gap.

## Dataset

This project uses the **UCI Internet Firewall Data** dataset. The original dataset contains 65,532 records and 12 columns: 11 input attributes and the target column `Action`. The target has four classes: `allow`, `deny`, `drop` and `reset-both`.

- Official dataset: https://archive.ics.uci.edu/dataset/542/internet%2Bfirewall%2Bdata
- DOI: `10.24432/C5131M`
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)

The notebook currently downloads the CSV from a public GitHub-hosted copy. The original dataset is credited to the UCI Machine Learning Repository. Please retain this attribution when reusing the dataset.

## Methodology

1. Load and inspect the firewall dataset.
2. Remove exact duplicate rows before splitting the data.
3. Convert the target labels to numeric values and handle invalid/missing feature values.
4. Split the cleaned data into 80% training and 20% testing sets using stratification.
5. Add duplicate records sampled with replacement from the training set only, at six levels: 0%, 10%, 25%, 50%, 75% and 100% of the original training-set size.
6. Train four classifiers and evaluate each model on the same untouched test set.
7. Repeat duplicate sampling with random seeds 42 and 123, then compare averaged results.

## Models

- Logistic Regression (with feature scaling)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (with feature scaling)

## Evaluation metrics

Accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score, multi-class ROC-AUC, generalization gap and training time. The notebook also generates classification reports, confusion matrices, feature-importance output and comparison plots.


## Repository structure

```text
impact-of-duplicate-data-on-ml-models/
|
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- COLAB_CELL_GUIDE.md
|
|-- notebooks/
|   |-- duplicate_data_experiment.ipynb
|
|-- outputs/
    |-- results/
    |-- visualizations/
    |-- confusion_matrices/
```


## How to run in Google Colab

1. Open `notebooks/duplicate_data_experiment.ipynb` in Google Colab.
2. Select **Runtime ➜ Run all**.
3. The notebook installs the required Python libraries, downloads the CSV, performs the experiment and saves the generated tables and figures in `project_outputs/`.
4. At the end, the notebook creates and downloads `mini_project_outputs.zip`.

You can also open the notebook directly from the GitHub repository using Colab's GitHub notebook picker.
Colab : https://colab.research.google.com/drive/1qEnlfCfG334GRXHnz8Oz2U7rdrD2Cz44

## Outputs

After execution, the notebook creates CSV files and visualizations, including:

- All experiment results and per-seed results
- Average results across the two duplicate-sampling seeds
- Classification reports
- Original, duplicated and deduplicated training-set comparison
- Random Forest feature importance
- Metric-versus-duplicate-level plots
- Baseline model comparison and class-wise performance charts
- Confusion matrices for baseline and 100%-duplicate experiments

Generated outputs are not committed by default. Add the selected final CSV files and plots to `outputs/` if you want to publish the results with the repository.

## Limitations

- The experiment uses one dataset.
- The same stratified train-test split is used throughout the comparison.
- Duplicate-sampling variability is represented using two random seeds.
- The target classes are imbalanced, so accuracy should be interpreted alongside macro F1 and class-wise results.

## License and attribution

The project code is provided under the MIT License in `LICENSE`. The UCI Internet Firewall Data dataset is separately licensed under CC BY 4.0; see the official UCI dataset page and DOI above. The project-code license does not replace or override the dataset's license.

