#!/usr/bin/env python
"""
====================================================================================
LUNG CANCER RISK PREDICTION DATASET — COLUMN REFERENCE
====================================================================================
Source: https://www.kaggle.com/datasets/dhrubangtalukdar/lung-cancer-prediction-dataset
Rows:   5,000
Cols:   30 (29 features + 1 target)
All columns are integer-encoded.
====================================================================================

COLUMN                   TYPE        RANGE       DESCRIPTION
------------------------------------------------------------------------------------
DEMOGRAPHICS
  age                    Continuous  18–90       Patient age in years
  gender                 Binary      0, 1        0 = Female, 1 = Male
  education_years        Continuous  5–20        Years of formal education
  income_level           Ordinal     1–5         Socioeconomic bracket (1=lowest, 5=highest)

SMOKING HISTORY
  smoker                 Binary      0, 1        Current/former smoker flag
  smoking_years          Continuous  0–52        Duration of smoking habit (years)
  cigarettes_per_day     Continuous  0–44        Average daily cigarette consumption
  pack_years             Continuous  0–60        (cigarettes_per_day / 20) × smoking_years
  passive_smoking        Binary      0, 1        Regular secondhand smoke exposure

ENVIRONMENTAL EXPOSURES
  air_pollution_index    Continuous  20–130      Local air quality index (higher = worse)
  occupational_exposure  Binary      0, 1        Workplace carcinogen exposure (dust, chemicals, etc.)
  radon_exposure         Binary      0, 1        Residential radon gas exposure

MEDICAL HISTORY
  family_history_cancer  Binary      0, 1        First-degree relative with cancer
  copd                   Binary      0, 1        Chronic Obstructive Pulmonary Disease diagnosis
  asthma                 Binary      0, 1        Asthma diagnosis
  previous_tb            Binary      0, 1        History of tuberculosis

SYMPTOMS (⚠ potential data leakage — may be consequences, not causes)
  chronic_cough          Binary      0, 1        Persistent cough present
  chest_pain             Binary      0, 1        Chest pain present
  shortness_of_breath    Binary      0, 1        Dyspnea present
  fatigue                Binary      0, 1        Chronic fatigue present

CLINICAL MEASUREMENTS
  bmi                    Continuous  16–37       Body Mass Index (kg/m²)
  oxygen_saturation      Continuous  85–100      SpO₂ percentage (normal ≥ 95%)
  fev1_x10               Continuous  5–37        Forced Expiratory Volume in 1s × 10
                                                 (divide by 10 for liters; normal ~3–4 L)
  crp_level              Continuous  0–33        C-Reactive Protein (mg/L); inflammation marker
  xray_abnormal          Binary      0, 1        Chest X-ray shows abnormality

LIFESTYLE
  exercise_hours_per_week Continuous 0–10        Weekly exercise hours
  diet_quality           Ordinal     1–5         Self-reported diet quality (1=poor, 5=excellent)
  alcohol_units_per_week Continuous  0–23        Weekly alcohol consumption (units)
  healthcare_access      Ordinal     1–5         Access to healthcare services (1=poor, 5=excellent)

TARGET
  lung_cancer_risk       Binary      0, 1        0 = Low risk, 1 = High risk

====================================================================================
NOTES:
- pack_years ≈ (cigarettes_per_day / 20) × smoking_years  → multicollinearity risk
- Symptom columns may reflect existing disease, not just risk factors (leakage)
- fev1_x10 is scaled by 10; divide by 10 for standard FEV1 in liters
- All values are integers; no missing data present in raw file
====================================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/lung_cancer.csv")
TARGET = "lung_cancer_risk"

# =========== EDA =============
print("======== Preview of Data ======== ")
print(f"Sample of the data:\n{df.head()}")
print(f"\n(Rows, Columns) = {df.shape}\n")
df.info()
print(f"\nDescribe (numeric data): {df.describe()}")
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")
print(f"\nNumber of Null Values: {df.isnull().sum().sum()}")

# Plotting



# ======== REMOVE REDUNCANCIES SO TO MAINTAIN INDEPENDENT PREDICTORS ====
# pack_years  = (cigs_per_day / 20) * smoking_years, so just keep pack_years
#df.drop(["smoking_years", "cigarettes_per_day", "passive_smoking"],
#        inplace=True, axis=1) # Do we want to keep 'passive_smoking' ?

# ======== Separate causes from consequences (Use for 2nd model!) ==========
# Potential leakage (could cause or result from disease)
leaks = ["chronic_cough", "chest_pain", "shortness_of_breath", 
         "fatigue", "oxygen_saturation", "fev1_x10", "crp_level", "xray_abnormal"]
# Remove the leaks
no_sympt_df = df.drop(leaks, axis=1)

# Stronger predictors will be those features available at intake, then
# lifestyle and socioeconomic features ...



breakpoint()






