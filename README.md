# ReLU and Tanh for breast cancer classification

**Deep Learning (CSC403-8) · CIA 1 – Component I A**  
Tejas Bhandarkar · Registration 2340173 · CHRIST (Deemed to be University), Bengaluru

The project compares two 30 → 16 → 8 → 1 networks on the Wisconsin Diagnostic Breast Cancer benchmark. Both use a sigmoid output. Model A uses ReLU in both hidden layers and Model B uses Tanh. This is an educational machine learning experiment, not a clinical diagnostic system.

## Review the work

| File or folder | Contents |
|---|---|
| `Deep_Learning_CIA1_Report.pdf` | Final formatted report with six figures, six tables and references |
| `Deep_Learning_CIA1_Report.docx` | Editable Word report with native equations |
| `Deep_Learning_CIA1_ANN_Comparison.ipynb` | Executed, self-contained notebook with code, training output, plots and interpretation |
| `ann_comparison.py` | Standalone experiment using the same functions as the notebook |
| `requirements.txt` | Tested package versions |
| `figures/` | Six publication-resolution PNG figures |
| `results/` | Machine-readable outputs listed below |

**Main finding.** On the shared 114-record test split, mean accuracy was 96.32% for ReLU and 96.14% for Tanh. ReLU had higher mean malignant recall (96.19% versus 93.33%), while Tanh had higher mean malignant precision (96.14% versus 94.01%). These are arithmetic means across five training seeds at a fixed threshold of 0.5.

## Install and run

The reference environment was Linux x86-64 using CPython 3.12.14, TensorFlow CPU 2.20.0 and Keras 3.15.1. Extract the ZIP and open a terminal inside `Deep_Learning_CIA1_Tejas_Bhandarkar`. On Linux with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ann_comparison.py --output rerun
```

This command rebuilds all ten models and writes fresh figures and CSVs in `rerun/`. With no `--output` argument, the script writes beside itself and replaces matching CSVs and PNGs. No dataset download, API key or model weights are needed after installation. The benchmark is bundled with scikit-learn.

The notebook already includes outputs. To rerun in a Jupyter interface, open it with a fresh Python kernel from this environment and choose **Restart Kernel and Run All Cells**. It runs without importing the standalone source. For command-line execution after environment activation:

```bash
python -m ipykernel install --sys-prefix --name cia1-ann --display-name "Python (CIA 1 ANN)"
jupyter execute --kernel_name=cia1-ann --inplace --timeout=1200 Deep_Learning_CIA1_ANN_Comparison.ipynb
```

Running the notebook regenerates the outputs in `figures/` and `results/` relative to its working directory. To preserve the distributed notebook and outputs, run it in a copy of the project folder.

## Dataset and controls

The original scikit-learn labels are 0 = malignant and 1 = benign. The project recodes malignant as 1 and benign as 0. There are 569 observations and 30 numeric features. Class counts are 357 benign and 212 malignant. [UCI dataset and DOI](https://doi.org/10.24432/C5DW2B) · [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

| Stage | Records | Procedure |
|---|---:|---|
| Training | 364 | Fit the scaler and update network parameters |
| Validation | 91 | Monitor end-of-epoch BCE, select and restore the minimum-loss checkpoint |
| Test | 114 | Evaluate all frozen models once per run with a prespecified threshold |

- Stratified outer split seed 2026. The stratified validation split within the remaining records uses seed 2027.
- Paired training seeds are 11, 23, 37, 53 and 71. Both members use identical seeded Glorot weights, zero biases and exactly the same sequence of training permutations, up to the point each stops.
- Adam uses a learning rate of 0.001, β₁ = 0.9, β₂ = 0.999 and epsilon-hat = 10⁻⁷. Binary cross-entropy is minimized in minibatches of 16, with the last partial batch retained.
- The maximum is 50 epochs and the patience is eight consecutive epochs without strictly improved validation BCE. The restored checkpoint is the lowest validation BCE even when the run reaches the cap.
- All ten models finish training before test inference. Architecture, threshold and representative seed 11 were fixed before scoring. Test probabilities feed only evaluation and visualization.
- Both class reports are saved. The summary precision, recall and F1 use malignant as positive. Accuracy uses both classes. ROC-AUC uses continuous scores. Sample standard deviations use a denominator of four.

## Machine-readable results

| File in `results/` | One row per | Meaning |
|---|---|---|
| `comparison_metrics.csv` | Activation | Mean and sample SD for the five test metrics |
| `per_run_metrics.csv` | Seed and activation | Metrics, confusion counts, epochs, validation loss and initial-weight hash |
| `test_predictions.csv` | Test record within a run | Original scikit-learn row index, recoded label, malignant score and hard prediction |
| `training_history.csv` | Epoch within a run | End-of-epoch full-training and validation accuracy and BCE |
| `classification_reports.csv` | Class or average within a run | Precision, recall, F1 and support for each class and the macro and weighted averages |
| `split_membership.csv` | Source row | Stratified partition and recoded class |

Per-class reports accompany the overall accuracy in `per_run_metrics.csv`. The row index in the exported files is a zero-based dataset position. It is never included in the model inputs. Figure filenames describe their contents. Report figures 1–3 use PNG files 01–03, figure 4 uses 06, figure 5 uses 04 and figure 6 uses 05.

## Verification and reproducibility limits

The standalone source and the notebook were executed independently. Every code cell ran in order in a fresh in-process Jupyter kernel. The two executions produced identical CSV and PNG files in this environment. Test probabilities were used to recompute confusion counts and metrics, and checkpoint epochs were checked against the training histories. The DOCX was rendered to PDF for page-by-page review.

CPU execution, single-thread TensorFlow operations, explicit initialization seeds and recorded row memberships reduce variability. Different processors, compilers or package versions may change floating-point operations and the best epoch. The five runs reuse the same test records. Their spread measures training-seed variability on one partition and does not quantify uncertainty under a change of patient population.

## Assessment coverage

| Evaluation criterion | Marks | Where to review |
|---|---:|---|
| Problem formulation and mathematical modelling | 2 | Report pp. 2–4. The notebook derives the neuron, activations, forward pass, BCE and gradients. |
| ANN design and implementation | 3 | Report pp. 3–6. The notebook and source show partitions, scaling, matched models, optimizer and checkpoint restoration. |
| Experimental analysis and performance evaluation | 3 | Report pp. 6–9. The CSVs retain seed-level scores, predictions and class reports. |
| Documentation, critical analysis and references | 2 | Report pp. 9–10 and this README. The notebook includes interpretation, limitations and source links. |

