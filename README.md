# Exploratory Data Analysis & Machine Learning on Titanic Dataset

This project applies **Exploratory Data Analysis (EDA)** and **Logistic Regression** to the classic Titanic dataset to predict passenger survival. It is designed as a clean, beginner-friendly machine learning project showcasing the complete workflow from data exploration to model evaluation.

---

## 📌 Objective

- Understand the Titanic dataset through comprehensive EDA
- Build a classification model to predict the **Survived** status
- Achieve an accuracy of approximately **81%** using Logistic Regression
- Demonstrate best practices in data preprocessing, visualization, and model evaluation

---

## 📊 Dataset

- **Dataset**: Titanic passenger survival data
- **Task**: Binary classification (Survived: 0 or 1)
- **Target Variable**: `Survived` (0 = Not Survived, 1 = Survived)
- **Key Features Used**:
  - `Pclass`: Passenger class (1, 2, or 3)
  - `Sex`: Gender (male/female)
  - `Age`: Passenger age
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Ticket fare
  - `Embarked`: Port of embarkation (S, C, Q)

Place the dataset file here: `data/titanic.csv`

---

## 🛠️ Tools & Technologies

**Language**: Python 3.x

**Libraries**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` & `seaborn` - Data visualization
- `scikit-learn` - Machine learning models and evaluation

**Environment**: Jupyter Notebook

---

## 📁 Project Structure

```
AI-ML-Assignment/
│
├── data/
│   └── titanic.csv           # Titanic dataset
│
├── notebooks/
│   └── EDA_and_Logistic_Regression.ipynb
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/cherry368/AI-ML-Assignment.git
cd AI-ML-Assignment
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Ensure `data/titanic.csv` exists with the standard Kaggle Titanic dataset format.

### 5. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/EDA_and_Logistic_Regression.ipynb
```

Execute all cells in order to reproduce the analysis and results.

---

## 📈 Modeling Details

### Data Preprocessing

1. **Handle Missing Values**:
   - `Age`: Filled with median value
   - `Embarked`: Filled with mode value

2. **Drop Irrelevant Columns**: `Name`, `Ticket`, `PassengerId`, `Cabin`

3. **Encode Categorical Variables**:
   - `Sex`: Binary mapping (male=0, female=1)
   - `Embarked`: One-hot encoding with drop_first=True

### Train-Test Split
- **Training set**: 80%
- **Test set**: 20%
- **Stratification**: Yes (preserves class distribution)

### Model
- **Algorithm**: Logistic Regression
- **Configuration**: `LogisticRegression(max_iter=1000)`

---

## 📊 Results & Interpretation

### Model Performance

- **Accuracy**: ~**0.81** (81%)
- **Confusion Matrix**:
  ```
  [[90 15]
   [19 55]]
  ```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Not Survived (0) | 0.83 | 0.86 | 0.84 | 105 |
| Survived (1) | 0.79 | 0.74 | 0.76 | 74 |
| **Accuracy** | | | **0.81** | **179** |
| Macro avg | 0.81 | 0.80 | 0.80 | 179 |
| Weighted avg | 0.81 | 0.81 | 0.81 | 179 |

### Key Insights

- The model achieves **81% accuracy** on the test set
- Performs slightly better at predicting **non-survivors** (86% recall) than survivors (74% recall)
- This imbalance is common in survival prediction due to class distribution in the dataset
- Female passengers had significantly higher survival rates
- Passenger class (Pclass) was a strong predictor of survival

---

## 🔍 Exploratory Data Analysis Highlights

### Visualizations Created

1. **Survival Count Plot**: Overall distribution of survivors vs. non-survivors
2. **Age Distribution**: Histogram showing age distribution with KDE
3. **Survival by Gender**: Countplot revealing gender-based survival differences

### Key EDA Findings

- **Gender Impact**: Women had a much higher survival rate (~74%) compared to men (~19%)
- **Age Distribution**: Most passengers were between 20-40 years old
- **Class Impact**: First-class passengers had higher survival rates than third-class
- **Missing Data**: Age had ~20% missing values; Embarked had minimal missing data

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

- Uses a simple Logistic Regression model without hyperparameter tuning
- Limited feature engineering (no interaction features or domain-specific features)
- No cross-validation employed
- Class imbalance not addressed
- Dataset is relatively small (sample data for demonstration)

### Potential Improvements

1. **Feature Engineering**:
   - Extract `Title` from passenger names
   - Create `FamilySize` from `SibSp` and `Parch`
   - Create `IsAlone` feature

2. **Model Enhancement**:
   - Try alternative models: Random Forest, Gradient Boosting, XGBoost
   - Perform hyperparameter tuning with GridSearchCV
   - Implement k-fold cross-validation

3. **Class Imbalance Handling**:
   - Use SMOTE or other resampling techniques
   - Adjust class weights in the model

4. **Advanced Analysis**:
   - Feature importance analysis
   - ROC-AUC curve and threshold optimization
   - Permutation feature importance

---

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib & Seaborn](https://matplotlib.org/)
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

---

## 🎓 Author

**Charan Kumar M V**
- GitHub: [@cherry368](https://github.com/cherry368)
- Location: Davangere, Karnataka, India
- Focus: Web Development, ML/AI Integration, Full-Stack Development

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Reach out via LinkedIn
- Email for professional inquiries

---

**Last Updated**: January 2026

⭐ If you find this project helpful, please consider giving it a star!
