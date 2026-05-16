# Cesarean Section Prediction using Maternal Health Data (NFHS-5)

A machine learning project that predicts whether a pregnancy will result in a cesarean section (C-section) using maternal demographic and health-related attributes from the **National Family Health Survey (NFHS-5)**.

> **Notebook by Vaibhav Aggarwal** — *University of Colorado, Boulder*
>
> **Presentation video:** https://youtu.be/aVArQxf_oGY

---

## > IMPORTANT: Dataset Download

> [!IMPORTANT]
> The data file **`IAIR7EFL_40000_non_null_caesarean.csv`** used in the main project file **`Project.ipynb`** is **too large to upload on GitHub**.
>
> **Kindly download it from the link below and place it inside the [`data/`](data/) directory before running the notebook:**
>
> Direct URL: `https://www.kaggle.com/datasets/vaibhavagg101/40000-individual-women-data-nfhs-5/`

---

## Introduction

This project aims to predict whether a pregnancy will result in a cesarean section using a dataset containing maternal demographic and health-related attributes. By analyzing factors such as age, antenatal care frequency, medical history, and lifestyle habits, the model assists healthcare providers in identifying pregnancies likely to require surgical intervention.

Cesarean section is a surgical procedure used to deliver a baby through incisions made in the abdomen and uterus. While life-saving in specific medical situations, unnecessary or elective cesarean deliveries carry substantial risks and long-term health implications, including infections, increased recovery times, and future pregnancy complications.

## Motivation

Accurate prediction of cesarean sections during pregnancy can significantly enhance maternal healthcare management in India through:

- **Timely Interventions in Rural and Semi-Urban Areas** - Early identification of high-risk pregnancies allows ASHA workers and local healthcare providers to intervene promptly.
- **Optimized Resource Allocation in Public Health Facilities** - Helps government hospitals efficiently allocate surgical teams, operation theatres, and blood banks.
- **Reduction of Unnecessary Cesarean Deliveries** - Avoids surgical risks, hospital stays, and financial burdens on families.
- **Enhanced Emergency Preparedness in Primary Health Centers (PHCs)** - Better preparation with supplies, skilled personnel, and emergency transportation.
- **Culturally Relevant Counseling and Community Support** - Addressing common fears and myths associated with cesarean deliveries.
- **Alignment with National Health Programs** - Supports initiatives like *Janani Suraksha Yojana (JSY)* and *Pradhan Mantri Surakshit Matritva Abhiyan (PMSMA)*.

## Task and Algorithms

- **Type of Task:** Supervised Learning — Binary Classification (predict cesarean: `1` or not: `0`)
- **Algorithms used:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forests
  - AdaBoost (Adaptive Boosting)

## Dataset

The dataset was obtained from the **National Family Health Survey (NFHS-5)** — specifically the individual women's dataset containing detailed responses from over 700,000 women across India. It was accessed through the International Institute for Population Sciences (IIPS) data catalog upon formal request.

The raw data was retrieved as `.DAT` and `.SPS` files, converted to `.CSV` using `pspp` on Linux, and filtered using `awk`, `grep`, and other shell utilities to extract **40,000 samples with non-null `cesarean_section` values**.

| Property         | Value                                               |
|------------------|-----------------------------------------------------|
| Size             | 485 MB (509,605,194 bytes)                          |
| Samples          | 40,000                                              |
| Features         | 5,972                                               |
| Data types       | 410 integers, 5,562 objects (later rectified)       |

**Example feature descriptions:**

- `M1` — Number of tetanus injections before birth
- `M2A` — Prenatal: doctor available before conception
- `ML1` — Number of times took Fansidar during pregnancy
- `S414` — Received antenatal care for pregnancy
- `S419E` — Abdomen examined during antenatal visit
- `S420B` — Told about pregnancy complication: convulsions

The repository also includes the NFHS questionnaire PDF (in [`data/`](data/)) which was used during surveying and is invaluable for understanding the context of features.

**Citation:** International Institute for Population Sciences (IIPS). (2021). *National Family Health Survey (NFHS-5), 2019-21: Individual Women's Data.* Retrieved from https://iipsdata.ac.in

## Project Workflow

1. **Data Cleaning** — Feature selection by domain knowledge, dropping irrelevant/high-NaN columns, data-type rectification.
2. **Feature Engineering** — Creating count features and custom scores (e.g., `past_c_section_count`, `macrosomia_count`, `care_advice_score`, `household_standards_score`, `media_exposure_score`).
3. **EDA** — Correlation matrices, histograms, VIF analysis for multicollinearity.
4. **Preprocessing** — Outlier removal via IQR, standardization (`StandardScaler`), imputation of unknown categories.
5. **Modeling** — `GridSearchCV` hyperparameter tuning, 5-fold cross-validation.
6. **Evaluation** — Accuracy, Precision, Recall, AUC, confusion matrices, feature importance.

After cleaning: dataset reduced from `40,000 × 5,972` (485 MB) to `36,199 × 45` (~104 MB).

## Results Summary

- **AdaBoost** emerged as the best-performing model across all key metrics (Accuracy, Precision, Recall, AUC).
- **Random Forest** delivered consistent and stable predictions with good generalization.
- **Logistic Regression** offered excellent interpretability but lacked predictive depth for complex patterns.
- **KNN** was less effective due to the curse of dimensionality.

Cross-validation confirmed the results were reproducible and not the result of overfitting.

## Required Libraries

For Linux systems, install the libraries with:

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn statsmodels notebook
```

If already installed, upgrade with:

```bash
pip install --upgrade pandas matplotlib seaborn numpy scikit-learn statsmodels
```

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/vaibhavagg101/C-Section_Prediction_ML.git
   cd C-Section_Prediction_ML
   ```
2. **Download the dataset** from the [Kaggle link above](#-important-dataset-download) and place `IAIR7EFL_40000_non_null_caesarean.csv` inside the [`data/`](data/) directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the notebook:
   ```bash
   jupyter notebook Project.ipynb
   ```

Then open the printed Jupyter URL in your browser.

## Project Structure

```
C-Section_Prediction_ML/
├── Project.ipynb          # Main analysis notebook
├── data/
│   ├── IAIR7EFL_40000_non_null_caesarean.csv  # <-- DOWNLOAD FROM KAGGLE
│   ├── IAIR7EFL.MAP                            # Column name mapping reference
│   ├── IAIR7EFL.DOCX                           # Dataset documentation
│   ├── column_mapping.csv                      # Generated column-name map
│   ├── pregnant_logo.png                       # Notebook logo
│   └── NFHS_Quest_2015_16_..._NFHS-4Womans.pdf # NFHS questionnaire
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build definition
└── README.md
```

## Key Takeaways

- **Data cleaning and preprocessing** are critical. No amount of sophisticated modeling can compensate for poor data quality.
- **Feature selection matters as much as model complexity**. Reducing ~6,000 features via domain knowledge, correlation, and VIF was essential.
- **Ensemble methods** (Random Forest, AdaBoost) outperformed simpler classifiers on this complex dataset.
- **Data limitations** like absence of clinical biomarkers (blood pressure, glucose, fetal monitoring) capped achievable accuracy.
- **Hyperparameter tuning** via `GridSearchCV` was a core component, not an optional enhancement.

## Future Improvements

- Enrich the dataset with objective clinical measurements (blood pressure, blood sugar, fetal weight estimates).
- Construct interaction terms and composite risk scores.
- Validate predictions with real-world clinical testing before any deployment.

## Author

**Vaibhav Aggarwal**

- GitHub: [@vaibhavagg101](https://github.com/vaibhavagg101)
- Project Repository: [C-Section_Prediction_ML](https://github.com/vaibhavagg101/C-Section_Prediction_ML)
- Presentation: [YouTube](https://youtu.be/aVArQxf_oGY)
