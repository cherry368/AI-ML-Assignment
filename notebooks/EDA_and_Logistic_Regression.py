"""
Titanic Dataset - Exploratory Data Analysis & Logistic Regression
Author: Charan Kumar M V
Date: January 2026

This script performs EDA and trains a Logistic Regression model on the Titanic dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure plotting style
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("Loading Titanic dataset...")
df = pd.read_csv("../data/titanic.csv")

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("Missing values after imputation:")
print(df.isnull().sum())

# Drop irrelevant columns
df = df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"])

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nDataset after preprocessing:")
print(df.head())

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
print(f"Non-survival rate: {(1 - df['Survived'].mean()):.2%}")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

# Prepare features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
