# CreditWise Loan System

A machine learning pipeline to predict loan approval decisions based on applicant financial and demographic data. The project covers end-to-end steps: data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model training, and evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Workflow](#workflow)
  - [1. Data Loading & Inspection](#1-data-loading--inspection)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Encoding](#4-encoding)
  - [5. Train-Test Split & Scaling](#5-train-test-split--scaling)
  - [6. Model Training & Evaluation (Before Feature Engineering)](#6-model-training--evaluation-before-feature-engineering)
  - [7. Feature Engineering](#7-feature-engineering)
  - [8. Model Training & Evaluation (After Feature Engineering)](#8-model-training--evaluation-after-feature-engineering)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Getting Started](#getting-started)
- [License](#license)

---

## Project Overview

The **CreditWise Loan System** is a supervised binary classification project. Given an applicant's income, credit score, employment status, loan purpose, and other attributes, the system predicts whether a loan should be **Approved** or **Not Approved**.

The pipeline compares three classical ML algorithms — before and after feature engineering — to measure the impact of engineered features on model performance.

---

## Dataset

**File:** `loan_approval_data.csv`

The dataset contains **162+ loan applications** with the following features:

| Column | Type | Description |
|---|---|---|
| `Applicant_ID` | Float | Unique applicant identifier (dropped before modeling) |
| `Applicant_Income` | Float | Monthly income of the primary applicant |
| `Coapplicant_Income` | Float | Monthly income of the co-applicant |
| `Employment_Status` | Categorical | Salaried / Self-employed / Contract / Unemployed |
| `Age` | Float | Age of the applicant |
| `Marital_Status` | Categorical | Married / Single |
| `Dependents` | Float | Number of dependents |
| `Credit_Score` | Float | Credit score of the applicant |
| `Existing_Loans` | Float | Number of existing active loans |
| `DTI_Ratio` | Float | Debt-to-Income ratio |
| `Savings` | Float | Total savings amount |
| `Collateral_Value` | Float | Value of collateral provided |
| `Loan_Amount` | Float | Requested loan amount |
| `Loan_Term` | Float | Loan term in months |
| `Loan_Purpose` | Categorical | Personal / Home / Car / Business / Education |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Education_Level` | Categorical | Graduate / Not Graduate |
| `Gender` | Categorical | Male / Female |
| `Employer_Category` | Categorical | Private / Government / MNC / Business / Unemployed |
| `Loan_Approved` | Categorical | **Target** — Yes / No |

> The dataset contains missing values in multiple columns which are handled during preprocessing.

---

## Project Structure

```
CreditWise_Loan_System p1/
│
├── creditWise_loan_system.ipynb   # Main Jupyter Notebook (full pipeline)
├── loan_approval_data.csv         # Raw dataset
├── CreditWise Loan System.pdf     # Project brief / problem statement
└── README.md                      # Project documentation (this file)
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical data visualizations |
| `scikit-learn` | ML models, preprocessing, and evaluation |

**Python version:** 3.x (Anaconda recommended)

---

## Workflow

### 1. Data Loading & Inspection

```python
df = pd.read_csv("loan_approval_data.csv")
df.head(10)
df.info()
df.describe()
df.isnull().sum()
```

- Loads the CSV dataset into a DataFrame.
- Inspects shape, data types, descriptive statistics, and missing value counts.

---

### 2. Preprocessing

**Identify column types:**
```python
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols   = df.select_dtypes(include=['float64']).columns
```

**Impute missing values:**
```python
# Numerical columns → fill with mean
num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Categorical columns → fill with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
```

---

### 3. Exploratory Data Analysis (EDA)

**Categorical distributions (Pie Charts)**

Eight categorical columns are plotted as pie charts in a 4-column grid layout:
- `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`
- `Education_Level`, `Gender`, `Employer_Category`, `Loan_Approved`

**Income & Credit Score distributions (Histogram + KDE)**

- `Applicant_Income` vs `Loan_Approved`
- `Coapplicant_Income` vs `Loan_Approved`
- `Credit_Score` vs `Loan_Approved`

**Feature distributions by target (Box Plots)**

Box plots for:
`Applicant_Income`, `Credit_Score`, `Dependents`, `DTI_Ratio`, `Savings` — split by `Loan_Approved`.

**Correlation Heatmap**

A 16×9 heatmap of the Pearson correlation matrix across all numerical features, annotated with values and using the `coolwarm` colour map.

---

### 4. Encoding

**Label Encoding** (ordinal/binary columns):
```python
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])
```

**One-Hot Encoding** (nominal columns, drop first to avoid multicollinearity):
```python
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded    = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols))
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
```

Also, `Applicant_ID` is dropped at this stage as it carries no predictive value.

---

### 5. Train-Test Split & Scaling

```python
X = df.drop(columns=['Loan_Approved'])
y = df['Loan_Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

- **80 / 20** train-test split with `random_state=42` for reproducibility.
- Features are standardized using `StandardScaler` (zero mean, unit variance).

---

### 6. Model Training & Evaluation (Before Feature Engineering)

Three models are trained and evaluated:

```python
models = [
    ("Logistic Regression", LogisticRegression()),
    ("KNN",                 KNeighborsClassifier(n_neighbors=5)),
    ("Naive Bayes",         GaussianNB())
]
```

For each model, the following metrics are reported:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

---

### 7. Feature Engineering

Three new features are derived to capture non-linear relationships:

| New Feature | Formula | Rationale |
|---|---|---|
| `DTI_Ratio_squared` | `DTI_Ratio ** 2` | Amplifies impact of high debt-to-income ratios |
| `Credit_Score_squared` | `Credit_Score ** 2` | Amplifies differences at higher credit score ranges |
| `Applicant_Income_log` | `log1p(Applicant_Income)` | Reduces right-skew of income distribution |

The original `DTI_Ratio`, `Credit_Score`, and `Applicant_Income` columns are dropped after engineering. The train-test split and scaling are repeated on the new feature set.

---

### 8. Model Training & Evaluation (After Feature Engineering)

The same three models are retrained and evaluated on the engineered feature set to measure improvement in predictive performance.

---

## Models Used

| Model | Description |
|---|---|
| **Logistic Regression** | Linear model for binary classification; interpretable and efficient |
| **K-Nearest Neighbours (KNN)** | Instance-based learner using `k=5` nearest neighbours |
| **Gaussian Naive Bayes** | Probabilistic model assuming feature independence and Gaussian distribution |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correct predictions over all predictions |
| **Precision** | Of all predicted positives, how many are truly positive |
| **Recall** | Of all actual positives, how many were correctly predicted |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Table of TP, FP, TN, FN counts |

---

## Results Summary

| Model | Stage | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Logistic Regression | Before FE | — | — | — | — |
| KNN | Before FE | — | — | — | — |
| Naive Bayes | Before FE | — | — | — | — |
| Logistic Regression | After FE | — | — | — | — |
| KNN | After FE | — | — | — | — |
| Naive Bayes | After FE | — | — | — | — |

> Run the notebook (`creditWise_loan_system.ipynb`) to populate the results table with actual values.

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or with conda:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/nisargpatel1906/creditWise_loan_system.git
   cd creditWise_loan_system
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook creditWise_loan_system.ipynb
   ```

3. Run all cells sequentially (`Kernel → Restart & Run All`).

---

## License

This project is for educational purposes. Feel free to use and adapt it with attribution.
