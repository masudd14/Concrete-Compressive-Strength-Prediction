<div align="center">

# 🏗️ Concrete Compressive Strength: High-Precision ML Benchmark

**A Domain-Informed Machine Learning Framework for Predicting High-Performance Concrete Compressive Strength (MPa) Using Abrams' Law and Advanced Tree Ensembles**

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/masudd14/Concrete-Compressive-Strength-Prediction/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Executive Summary](#-executive-summary)
- [Civil Engineering Motivation](#-civil-engineering-motivation)
- [Dataset Architecture](#-dataset-architecture)
- [Domain-Informed Concrete Science Feature Engineering](#-domain-informed-concrete-science-feature-engineering)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Experimental Benchmarking (11 Models)](#-experimental-benchmarking-11-models)
  - [Comprehensive Performance Leaderboard](#comprehensive-performance-leaderboard)
  - [Key Findings & Model Analysis](#key-findings--model-analysis)
- [Repository Structure](#-repository-structure)
- [Quickstart & Usage](#-quickstart--usage)
  - [1. Environment Setup](#1-environment-setup)
  - [2. CLI Mixture Prediction](#2-cli-mixture-prediction)
  - [3. Running Unit Tests](#3-running-unit-tests)
  - [4. Jupyter Notebook Research](#4-jupyter-notebook-research)
- [Citation & License](#-citation--license)

---

## 🌟 Executive Summary

Concrete compressive strength is the cornerstone parameter in structural and civil engineering design. Standard experimental evaluation requires preparing standardized cylindrical or cubical specimens and waiting for **28 days of curing** under controlled humidity and temperature. 

This repository develops a production-grade machine learning system that models the highly non-linear chemical and mechanical hydration interactions between mix constituents, predicting 28-day and multi-age compressive strength **instantaneously** with:
* **$R^2$ Score: 0.943+** on completely unseen holdout test mixtures.
* **Root Mean Squared Error (RMSE): ~4.21 MPa** (~11.8% error relative to mean concrete strength).
* **Mean Absolute Error (MAE): ~2.60 MPa**.

---

## 🏛️ Civil Engineering Motivation

In modern concrete technology (High-Performance Concrete - HPC), the mixture design is no longer a simple ternary blend of cement, water, and aggregates. Industrial concrete incorporates supplementary cementitious materials (SCMs):
* **Blast Furnace Slag:** Enhances long-term durability and resistance to chemical attack.
* **Fly Ash:** Improves workability and reduces heat of hydration in mass concrete pours.
* **Superplasticizers (High-Range Water Reducers):** Drastically lowers the required water content while maintaining fluidity.

Because the hydration kinetics of multi-binder systems exhibit severe non-linearities and age-dependent gain, classical empirical formulas are insufficient. Machine learning enables rapid optimization of mix designs, reducing carbon emissions by maximizing cement replacement while guaranteeing structural safety.

---

## 📊 Dataset Architecture

Experiments are conducted on the standardized **UCI Concrete Compressive Strength Dataset** (donated by Prof. I-Cheng Yeh, Chung-Hua University).

* **Samples:** 1,030 unique concrete mixture formulations.
* **Input Features:** 8 quantitative mixture ingredients and curing age.
* **Target:** Compressive strength in Megapascals ($\text{MPa}$).

### Feature Definitions & Physical Units

| Attribute | Role | Physical Unit | Description & Impact on Strength |
| :--- | :---: | :---: | :--- |
| **Cement** | Input | $\text{kg/m}^3$ | Primary hydraulic binder providing initial and continuous strength |
| **Blast Furnace Slag** | Input | $\text{kg/m}^3$ | Latent hydraulic byproduct of iron smelting; boosts late strength |
| **Fly Ash** | Input | $\text{kg/m}^3$ | Pozzolanic coal combustion byproduct; improves workability |
| **Water** | Input | $\text{kg/m}^3$ | Hydration medium; excess water causes porosity and reduces strength |
| **Superplasticizer** | Input | $\text{kg/m}^3$ | Polycarboxylate or naphthalene ether; disperses cement agglomerates |
| **Coarse Aggregate** | Input | $\text{kg/m}^3$ | Crushed stone ($> 4.75\text{ mm}$); forms the skeletal framework |
| **Fine Aggregate** | Input | $\text{kg/m}^3$ | Natural sand ($< 4.75\text{ mm}$); fills voids between coarse aggregate |
| **Age** | Input | $\text{Days}$ | Curing time elapsed prior to uniaxial compression testing (1 to 365 days) |
| **Compressive Strength** | **Target** | $\text{MPa}$ | Ultimate load capacity per unit area at failure |

---

## 🔬 Domain-Informed Concrete Science Feature Engineering

Unlike black-box modeling on raw measurements, this project injects foundational civil engineering principles into the feature space through the `ConcreteFeatureEngineering` pipeline:

1. **Total Binder Mass ($B$):**
   $$\text{Total Binder} = \text{Cement} + \text{Blast Furnace Slag} + \text{Fly Ash}$$
   Combines all active hydraulic and pozzolanic components into the total cementitious matrix.

2. **Abrams' Water-to-Cement Ratio ($w/c$):**
   $$\text{Water/Cement} = \frac{\text{Water}}{\text{Cement}}$$
   Formulated by Duff A. Abrams (1918), establishing that concrete strength is inversely related to the water-to-cement ratio.

3. **Water-to-Binder Ratio ($w/b$):**
   $$\text{Water/Binder} = \frac{\text{Water}}{\text{Total Binder}}$$
   The modern generalization of Abrams' Law for blended cement systems.

4. **Aggregate Interlocking Ratio:**
   $$\text{Aggregate Ratio} = \frac{\text{Coarse Aggregate}}{\text{Fine Aggregate}}$$
   Measures aggregate packing density, skeletal stiffness, and mortar volume.

5. **Superplasticizer-to-Binder Ratio:**
   $$\text{SP Ratio} = \frac{\text{Superplasticizer}}{\text{Total Binder}}$$
   Quantifies the dosage level relative to total binder surface area.

6. **Logarithmic Curing Age ($\ln(1 + \text{Age})$):**
   Because cement hydration kinetics follow a logarithmic deceleration over time (rapid gain in days 1–7, moderate gain in days 7–28, asymptotic plateau beyond 28 days), log-transforming age linearizes temporal strength development.

---

## 🔄 Machine Learning Workflow

```
Raw Concrete Mix (8 Features)
   │
   ├──> 1. Domain Feature Engineering (src/features.py)
   │       └── Adds Abrams Law, Binder Ratios, Log Age (14 Features)
   │
   ├──> 2. Feature Normalization (StandardScaler)
   │
   ├──> 3. Multi-Model Benchmark (11 Classical Regressors)
   │       ├── Linear: LinearRegression, ElasticNetCV
   │       ├── Kernel: Support Vector Regression (SVR - RBF)
   │       ├── Metric: K-Nearest Neighbors (KNN)
   │       └── Ensembles: Random Forest, Extra Trees, GBR, HGB, XGBoost, CatBoost
   │
   ├──> 4. Optimal Weighted Ensemble (VotingRegressor)
   │       └── GBR (3) + XGB (3) + CatBoost (3) + ExtraTrees (2) + RF (1)
   │
   └──> 5. Evaluation on Unseen Test Split ──> R²: 0.94+, RMSE: 4.21 MPa
```

---

## 📈 Experimental Benchmarking (11 Models)

### Comprehensive Performance Leaderboard

All models were evaluated using 5-fold cross-validation and tested against identical unseen holdout evaluation and test partitions:

| Rank | Model | Model Family | Test $R^2$ Score | Test RMSE ($\text{MPa}$) | Test MAE ($\text{MPa}$) | Test Error ($\%$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **CatBoost** | Gradient Boosting | **0.9434** | **4.218** | 2.748 | **11.85%** |
| 🥈 | **Gradient Boosting (GBR)** | Gradient Boosting | **0.9433** | **4.221** | **2.595** | **11.86%** |
| 🥉 | **XGBoost** | Gradient Boosting | **0.9426** | 4.247 | 2.708 | 11.94% |
| 4 | **HistGradientBoosting** | Histogram Boosting | 0.9422 | 4.263 | 2.526 | 11.98% |
| 5 | **Voting Ensemble** | Weighted Bagging/Boosting | 0.9415 | 4.287 | 2.671 | 12.05% |
| 6 | **Extra Trees** | Randomized Forests | 0.9246 | 4.866 | 2.852 | 13.68% |
| 7 | **Random Forest** | Bagging Ensemble | 0.9086 | 5.359 | 3.572 | 15.06% |
| 8 | **K-Nearest Neighbors (KNN)**| Metric Instance-Based | 0.9070 | 5.405 | 3.460 | 15.19% |
| 9 | **Support Vector Regression (SVR)** | Kernel SVM (RBF) | 0.9004 | 5.594 | 3.716 | 15.72% |
| 10 | **Linear Regression** | Ordinary Least Squares | 0.8290 | 7.330 | 5.677 | 20.60% |
| 11 | **ElasticNetCV** | $L_1 / L_2$ Regularized | 0.8238 | 7.441 | 5.709 | 20.92% |

### Key Findings & Model Analysis

1. **Superiority of Gradient Boosted Trees:** CatBoost, Scikit-learn GBR, and XGBoost demonstrated outstanding predictive fidelity ($R^2 > 0.94$), effectively capturing the sharp threshold behaviors where micro-silica and superplasticizers interact.
2. **Impact of Domain Features:** Introducing Abrams' ratio and logarithmic age boosted linear model baselines from below $0.70$ up to $0.82$, and propelled tree ensembles past $0.94$.
3. **Generalization:** The test set error remains stable at ~11.8%, aligning closely with the inherent experimental variability of ASTM C39 physical compression testing.

---

## 📁 Repository Structure

```text
Concrete-Compressive-Strength-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated GitHub Actions test pipeline
├── data/
│   └── raw/
│       ├── Concrete_Data.csv          # Standardized CSV version (1,030 samples)
│       ├── Concrete_Data.xls          # Original UCI Excel sheet
│       └── Concrete_Readme.txt        # UCI attribute codebook
├── notebooks/
│   └── 01_concrete_strength_ml.ipynb  # Comprehensive EDA, feature engineering & model benchmarking
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Clean data ingestion and feature/target splitting
│   ├── features.py                    # ConcreteFeatureEngineering transformer (Abrams law)
│   ├── pipeline.py                    # End-to-end production scikit-learn Pipeline
│   └── evaluate.py                    # MAE, RMSE, R2, and percentage error metrics
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py               # Automated unit tests for feature math & metrics
├── .gitignore                         # Python, cache, and OS exclusions
├── CONTRIBUTING.md                    # Guidelines for contributors
├── LICENSE                            # MIT Open Source License
├── pyproject.toml                     # Standard Python packaging configuration
├── README.md                          # Master documentation
├── requirements.txt                   # Complete dependencies
└── sample_predict.py                  # Standalone CLI tool for predicting concrete mix strength
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/masudd14/Concrete-Compressive-Strength-Prediction.git
cd Concrete-Compressive-Strength-Prediction

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. CLI Mixture Prediction

Predict compressive strength for any concrete mix design via the command line:

```bash
python sample_predict.py --cement 350 --slag 50 --fly_ash 0 --water 160 --sp 6.0 --coarse 980 --fine 750 --age 28
```

Output:
```text
=======================================================
 🏗️ CONCRETE COMPRESSIVE STRENGTH PREDICTION
=======================================================
 Cement:           350.0 kg/m³
 Blast Slag:       50.0 kg/m³
 Fly Ash:          0.0 kg/m³
 Water:            160.0 kg/m³
 Superplasticizer: 6.0 kg/m³
 Coarse Agg.:      980.0 kg/m³
 Fine Agg.:        750.0 kg/m³
 Age:              28 days
-------------------------------------------------------
 PREDICTED STRENGTH: 52.14 MPa
=======================================================
```

### 3. Running Unit Tests

Run automated test suites verifying mathematical calculations and schema integrity:

```bash
python -m unittest discover -s tests
```

### 4. Jupyter Notebook Research

Step through the interactive visualizations, correlation heatmaps, residual analyses, and model comparisons:

```bash
jupyter notebook notebooks/01_concrete_strength_ml.ipynb
```

---

## 📜 Citation & License

* **Dataset Citation:** Yeh, I-Cheng (1998). "Modeling of strength of high-performance concrete using artificial neural networks." *Cement and Concrete Research*, Vol. 28, No. 12, pp. 1797-1808.
* **License:** Distributed under the [MIT License](LICENSE).
