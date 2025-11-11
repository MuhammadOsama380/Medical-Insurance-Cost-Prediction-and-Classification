# Medical Insurance Cost Prediction and Classification

**Author:** Muhammad Osama  
**Course:** TensorFlow & Keras with Python (Fanshawe College)  
**Institution:** Fanshawe College  
**LinkedIn:** [muhammad-osama-872328202](https://www.linkedin.com/in/muhammad-osama-872328202)  
**GitHub:** [MuhammadOsama380](https://github.com/MuhammadOsama380)

---

## Project Overview
This project predicts individual medical insurance charges and classifies patients as **high-cost** or **low-cost** using both **Machine Learning** and **Deep Learning** approaches.  
The objective is to enhance insurance pricing accuracy and help insurers identify high-risk individuals more effectively.

The project combines:
- Regression (continuous cost prediction)
- Classification (binary identification of above-median cost patients)
- Performance benchmarking between **baseline models (Linear/Logistic Regression)** and **Deep Neural Networks (TensorFlow/Keras)**

---

## Dataset Overview
- **Source:** [Kaggle - Medical Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)  
- **Records:** 1,338  
- **Features:** 7 total  
  - **Numerical:** age, bmi, children, charges  
  - **Categorical:** sex, smoker, region  
- **Key Stats:**
  - Median charges: **$13,270**  
  - Age range: **18–64 years**  
  - Smokers incur ~3.8× higher charges on average  

---

## Problem Statement
Insurance companies face difficulty accurately predicting health costs and identifying high-risk clients due to complex non-linear data patterns.  
This project addresses:
1. Predicting **insurance cost** using regression.  
2. Classifying **high-cost patients** (charges > median) using deep learning.

---

## Methodology

### 1. Data Preprocessing
- Label Encoding: Converted categorical variables (`sex`, `smoker`, `region`) into numeric form.  
- Feature Scaling: Standardized numerical variables using `StandardScaler`.  
- Feature Engineering: Created a binary target variable `high_cost` based on median charges.  
- Data Split: 80% training / 20% testing.

### 2. Exploratory Data Analysis (EDA)
- Age shows a bimodal distribution (20s and 50s peaks).  
- Charges are right-skewed (max: $63,770).  
- Smokers and BMI strongly influence costs.  
- Correlations:
  - Age ↔ Charges: 0.30  
  - BMI ↔ Charges: 0.20  

### 3. Baseline Models
| Model | Task | R² / Accuracy | MSE | MAE | F1 Score |
|--------|------|---------------|-----|-----|-----------|
| Linear Regression | Regression | 0.7833 | 33,635,210 | 4,186 | — |
| Logistic Regression | Classification | 91% | — | — | 0.904 |

### 4. Deep Learning Models (TensorFlow/Keras)
**Regression DNN:**
- Layers: 64 → Dropout(0.3) → 32 → Dropout(0.2) → Output(1)
- Optimizer: Adam  
- Loss: MSE  
- Metrics: MAE, R²  
- Early stopping to prevent overfitting  

**Results:**
- R² = 0.8002, MSE = 31,022,003, MAE = 3,843

**Classification DNN:**
- Layers: 64 → Dropout(0.3) → 32 → Dropout(0.2) → Sigmoid  
- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy, Precision, Recall, F1  

**Results:**
- Accuracy = 95.9%, Precision = 98.3%, F1 = 95.4%

---

## K-Fold Cross Validation (5-Fold)
| Model | Metric | Mean ± Std |
|--------|---------|-------------|
| DNN (Regression) | MSE | 36.7M ± 3.6M |
| DNN (Regression) | MAE | 4,110 ± 144 |
| DNN (Classification) | Accuracy | 94.2% ± 2% |
| DNN (Classification) | F1 Score | 94.0% ± 1.9% |

Cross-validation confirmed both regression and classification DNNs are stable and generalize well across folds.

---

## Comparative Performance
| Task | Baseline | DNN | Improvement |
|------|-----------|-----|--------------|
| Regression (R²) | 0.7833 | 0.8002 | +2.1% |
| Regression (MAE) | 4,186 | 3,843 | -8.2% |
| Classification (Accuracy) | 91.0% | 95.9% | +4.9% |
| Classification (Precision) | 88.2% | 98.3% | +10.1% |

---

## Visualizations
- EDA plots for age, BMI, and charge distributions  
- Heatmap of feature correlations  
- DNN training curves (loss & accuracy)  
- Classification confusion matrix  
- Regression residual scatterplots  

---

## How to Run
```bash
# Clone this repository
git clone https://github.com/MuhammadOsama380/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook Muhammad_Osama_FinalProject_Code.ipynb
```

---

## Future Enhancements
- Automate hyperparameter tuning using Bayesian optimization  
- Apply SHAP/LIME for model interpretability  
- Introduce residual or CNN architectures for enhanced learning  
- Expand dataset with temporal or medical history features  
- Build interactive dashboards for insurer visualization  

---

## Tools & Libraries
- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas / NumPy / Matplotlib / Seaborn  

---

## Repository Structure
```
Medical-Insurance-Cost-Prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── Muhammad_Osama_FinalProject_Code.ipynb
│
├── reports/
│   └── Muhammad_Osama_FinalProject_Report.pdf
│
├── presentation/
│   └── Muhammad_Osama_FinalProject_Presentation.pdf
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Developed by **Muhammad Osama** as part of the **TensorFlow & Keras with Python** course at **Fanshawe College**.  
If you found this project helpful, please ⭐ the repository or connect with me on [LinkedIn](https://www.linkedin.com/in/muhammad-osama-872328202).
