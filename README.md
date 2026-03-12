<div align="center">

<!-- Animated Banner -->
<h1 style="background: linear-gradient(45deg, #FF6B6B, #FF8E72, #FFA559, #FFD93D, #6BCB77, #4D96FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3em; font-weight: bold; animation: gradient 3s ease infinite;">
  ❤️ Heart Disease Prediction Model
</h1>

<!-- Description Banner -->
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
  <p style="color: white; font-size: 1.1em; font-weight: 500; margin: 0;">
    🔬 Advanced Machine Learning Model for Predictive Healthcare Analysis
  </p>
</div>

---

</div>

## 📊 Project Overview

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 8px; padding: 20px; margin: 15px 0;">

**Heart Disease Prediction** is a comprehensive machine learning project that leverages modern data science techniques to predict cardiovascular disease presence based on clinical parameters. This project demonstrates the complete data science pipeline from exploration to model deployment.

</div>

---

## 🎯 Project Objectives

<table style="width: 100%; border-collapse: collapse;">
  <tr style="background-color: #4D96FF;">
    <td style="border: 2px solid #2D5FFF; padding: 15px; color: white; font-weight: bold; width: 33%;">📈 Data Analysis</td>
    <td style="border: 2px solid #2D5FFF; padding: 15px; color: white; font-weight: bold; width: 33%;">🤖 Model Training</td>
    <td style="border: 2px solid #2D5FFF; padding: 15px; color: white; font-weight: bold; width: 33%;">✅ Validation</td>
  </tr>
  <tr>
    <td style="border: 2px solid #B0D4FF; padding: 15px;">Explore and visualize 13+ clinical variables for comprehensive understanding</td>
    <td style="border: 2px solid #B0D4FF; padding: 15px;">Implement Logistic Regression for binary classification task</td>
    <td style="border: 2px solid #B0D4FF; padding: 15px;">Evaluate model performance with test/train split methodology</td>
  </tr>
</table>

---

## 📁 Project Structure

```
📦 Heart-Disease-Prediction
│
├── 📄 README.md                 ← You are here
├── 📊 dataset.csv               ← Dataset with 303 records
├── 📓 TASK_5.ipynb              ← Complete analysis notebook
│
├── 📁 anaconda_projects/
│   ├── 📁 db/                   ← Database configurations
│   └── 📄 Project metadata
│
├── 📁 .ipynb_checkpoints/       ← Jupyter checkpoints
├── 🔒 .gitignore               ← Git configuration
└── 📁 .git/                     ← Version control

```

---

## 📋 Dataset Information

<div style="background: linear-gradient(90deg, #FFD93D, #FFA559); border-radius: 8px; padding: 20px; margin: 15px 0;">

### Clinical Parameters Analyzed

| Parameter | Type | Description |
|-----------|------|-------------|
| **age** | Numeric | Age in years |
| **sex** | Binary | 1 = Male, 0 = Female |
| **cp** | Categorical | Chest pain type (0-3) |
| **trestbps** | Numeric | Resting blood pressure (mmHg) |
| **chol** | Numeric | Serum cholesterol (mg/dl) |
| **fbs** | Binary | Fasting blood sugar > 120 mg/dl |
| **restecg** | Categorical | Resting electrocardiographic (0-2) |
| **thalach** | Numeric | Maximum heart rate achieved |
| **exang** | Binary | Exercise-induced angina |
| **oldpeak** | Numeric | ST depression |
| **slope** | Categorical | Slope of peak ST segment |
| **ca** | Numeric | Number of major vessels |
| **thal** | Categorical | Thalassemia type |
| **target** | Binary | 1 = Disease, 0 = No Disease |

**Dataset Size:** 303 records | **Features:** 13 | **Missing Values:** 0

</div>

---

## 🛠️ Technology Stack

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">

<div style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 8px; padding: 15px; color: white;">
  <strong>Core Libraries</strong>
  <ul>
    <li>Python 3.x</li>
    <li>Pandas - Data manipulation</li>
    <li>NumPy - Numerical computing</li>
    <li>Scikit-learn - ML algorithms</li>
  </ul>
</div>

<div style="background: linear-gradient(135deg, #f093fb, #f5576c); border-radius: 8px; padding: 15px; color: white;">
  <strong>Visualization & Analysis</strong>
  <ul>
    <li>Matplotlib - Static plots</li>
    <li>Jupyter Notebook - Interactive environment</li>
    <li>Anaconda - Environment management</li>
  </ul>
</div>

</div>

---

## 🚀 Getting Started

### Prerequisites

```bash
# Ensure you have Python 3.7+ installed
python --version

# Required packages
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter Notebook
jupyter notebook
```

---

## 📊 Key Findings & Results

<div style="background: linear-gradient(90deg, #6BCB77, #4D96FF); border-radius: 8px; padding: 20px; margin: 15px 0; color: white;">

### Model Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Train Set Size** | 242 records (80%) | ✅ |
| **Test Set Size** | 61 records (20%) | ✅ |
| **Algorithm** | Logistic Regression | ✅ |
| **Validation Method** | Train/Test Split | ✅ |

</div>

---

## 💻 Implementation Details

### 1️⃣ Data Preparation

<div style="background: #F0F4FF; border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; border-radius: 5px;">

```python
# Import essential libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('dataset.csv')

# Split into training and testing sets
train_set, test_set = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42
)
```

</div>

**Purpose:** Ensure unbiased model evaluation by separating data into training (80%) for learning patterns and testing (20%) for performance validation.

---

### 2️⃣ Model Training

<div style="background: #FFF0F4; border-left: 4px solid #f5576c; padding: 15px; margin: 10px 0; border-radius: 5px;">

```python
# Import model
from sklearn.linear_model import LogisticRegression

# Prepare features and labels
feature = train_set.drop('target', axis=1)
labels = train_set['target']

# Create and train model
model = LogisticRegression()
model.fit(feature, labels)
```

</div>

**Why Logistic Regression?** Excellent for binary classification, interpretable coefficients, and fast training on medical datasets.

---

### 3️⃣ Model Evaluation

<div style="background: #F0FFF4; border-left: 4px solid #6BCB77; padding: 15px; margin: 10px 0; border-radius: 5px;">

```python
# Make predictions on test set
feature_test = test_set.drop('target', axis=1)
labels_test = test_set['target']

predictions = model.predict(feature_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predictions)
```

</div>

---

## 📈 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset (303 records)  ➜  Data Exploration  ➜  Preprocessing │
│         ↓                        ↓                     ↓         │
│    (dataset.csv)          (df.info(), describe)    (Encoding)   │
│                                                                 │
│  Train/Test Split (80/20)  ➜  Model Training  ➜  Evaluation   │
│         ↓                        ↓                     ↓         │
│  242/61 records      Logistic Regression    Accuracy Metrics    │
│                                                                 │
│                      ➜ Predictions ➜ Insights                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Engineering

<div style="background: linear-gradient(135deg, #FFD93D, #FFA559); border-radius: 8px; padding: 20px; margin: 15px 0;">

### Clinical Correlations

**High Impact Features:**
- Maximum heart rate achieved (thalach)
- Chest pain type (cp)
- ST depression (oldpeak)
- Number of major vessels (ca)

**Medium Impact Features:**
- Age, Sex, Blood Pressure, Cholesterol
- Exercise-induced angina
- Resting ECG results

**Model Interpretation:** Coefficients indicate relative importance of each feature in disease prediction.

</div>

---

## 🎓 Learning Outcomes

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: linear-gradient(90deg, #667eea, #764ba2);">
    <th style="border: 2px solid #667eea; padding: 12px; color: white;">Concept</th>
    <th style="border: 2px solid #667eea; padding: 12px; color: white;">What You'll Learn</th>
  </tr>
  <tr>
    <td style="border: 2px solid #B0C4FF; padding: 12px; background: #F0F4FF;"><strong>Train/Test Split</strong></td>
    <td style="border: 2px solid #B0C4FF; padding: 12px; background: #F0F4FF;">Importance of data partition in preventing overfitting</td>
  </tr>
  <tr>
    <td style="border: 2px solid #B0C4FF; padding: 12px;"><strong>Logistic Regression</strong></td>
    <td style="border: 2px solid #B0C4FF; padding: 12px;">Binary classification fundamentals and interpretation</td>
  </tr>
  <tr>
    <td style="border: 2px solid #B0C4FF; padding: 12px; background: #F0F4FF;"><strong>Model Validation</strong></td>
    <td style="border: 2px solid #B0C4FF; padding: 12px; background: #F0F4FF;">Performance metrics and real-world evaluation</td>
  </tr>
  <tr>
    <td style="border: 2px solid #B0C4FF; padding: 12px;"><strong>Healthcare ML</strong></td>
    <td style="border: 2px solid #B0C4FF; padding: 12px;">Application of ML in medical diagnostics</td>
  </tr>
</table>

---

## 🔄 Usage Instructions

### Running the Complete Analysis

```bash
# Option 1: Jupyter Notebook (Interactive)
jupyter notebook TASK_5.ipynb

# Option 2: Python Script (Automated)
python train_model.py
```

### Making Predictions

```python
# Load the trained model
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Prepare new data
new_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [1],
    # ... other features
})

# Predict
prediction = model.predict(new_data)
print(f"Disease Risk: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
```

---

## 🐛 Troubleshooting

<div style="background: #FFF3CD; border-left: 4px solid #FF9800; padding: 15px; margin: 15px 0; border-radius: 5px;">

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` |
| **Jupyter not found** | Execute `pip install jupyter` |
| **Dataset not loading** | Verify `dataset.csv` is in project root |
| **Memory issues** | Use `pd.read_csv(..., chunksize=100)` for large files |

</div>

---

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas User Guide](https://pandas.pydata.org/docs)
- [Healthcare ML Best Practices](https://en.wikipedia.org/wiki/Machine_learning_in_healthcare)
- [Logistic Regression Deep Dive](https://en.wikipedia.org/wiki/Logistic_regression)

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** changes (`git commit -m 'Add feature'`)
4. **Push** to branch (`git push origin feature/improvement`)
5. **Submit** a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Elevate Lab - Task 5 Project**
- Institution: Elevate Lab January Cohort
- Focus: Machine Learning for Healthcare
- Year: 2026

---

## 📧 Contact & Support

<div style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; color: white;">

**Questions or Feedback?**

Open an issue on GitHub or submit a discussion thread. We actively maintain this project and welcome all inquiries.

</div>

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star!**

Made with passion for healthcare and machine learning.

</div>

---

### Version History

- **v1.0** (Feb 2026) - Initial release with Logistic Regression model
- Documentation: Complete and comprehensive
- Status: Active Development
  . .. .... 
