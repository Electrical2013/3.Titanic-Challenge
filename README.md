# 🚢 Titanic Challenge — Survival Prediction Using Machine Learning  

### 📘 Project Overview  
This project applies data science and machine learning techniques to predict passenger survival on the Titanic using the classic **Kaggle Titanic dataset**.  
It demonstrates a complete end-to-end ML workflow — from data acquisition and cleaning to feature engineering, model building, and evaluation.

---

## 🎯 Objectives  
- Analyze historical Titanic passenger data to uncover survival trends  
- Engineer and preprocess features to enhance model learning  
- Train and evaluate multiple ML algorithms  
- Compare model performance and select the best performer  
- Visualize and interpret results effectively  

---

## 📂 Dataset Description  
Dataset Source: [Kaggle Titanic Machine Learning Challenge](https://www.kaggle.com/c/titanic)  
**Key Features:**
| Feature | Description |
|----------|--------------|
| `PassengerId` | Unique ID assigned to each passenger | Integer |
| `Survived` | Target variable — 0 = Did not survive, 1 = Survived | Integer (Binary) |
| `Pclass` | Passenger ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Categorical (Ordinal) |
| `Name` | Passenger’s full name (includes title) | Text |
| `Sex` | Gender of the passenger | Categorical |
| `Age` | Age in years | Float |
| `SibSp` | Number of siblings/spouses aboard | Integer |
| `Parch` | Number of parents/children aboard | Integer |
| `Ticket` | Ticket number (often alphanumeric) | Text |
| `Fare` | Passenger fare (ticket price) | Float |
| `Cabin` | Cabin number (many missing values) | Text |
| `Embarked` | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) | Categorical |

---

## 🧭 Methodology  
1. **Data Cleaning** – Removed irrelevant or null-heavy columns (e.g., `Cabin`), filled missing values for `Age`, `Embarked`, and `Fare`.  
2. **Exploratory Data Analysis (EDA)** – Identified trends showing higher survival for females and 1st-class passengers.  
3. **Feature Engineering** – Created new variables such as `family_size` and `IsAlone` to improve predictive power.  
4. **Model Development** – Implemented models like **Decision Tree**, **Logistic Regression**, **Random Forest**, and **XGBoost**.  
5. **Evaluation & Optimization** – Used accuracy and cross-validation to assess and refine model performance.  
6. **Visualization** – Used `matplotlib` and `seaborn` for distribution, survival trends, and feature-importance plots.

---

## 🧩 Machine Learning Project Life Cycle & Code Highlights  

| Stage | Key Code | Explanation |
|-------|-----------|-------------|
| **1️⃣ Problem Definition** | `Goal: Predict survival (binary classification)` | Defined task objective |
| **2️⃣ Data Collection** | `pd.read_csv("train.csv")` | Loaded train/test/gender CSVs |
| **3️⃣ Data Inspection** | `.info()`, `.head()` | Checked structure and missing values |
| **4️⃣ Cleaning** | `drop('Cabin')`, `fillna()` | Removed or imputed nulls |
| **5️⃣ Feature Engineering** | Created `family`, `family_size` | Captured group survival patterns |
| **6️⃣ Encoding** | `pd.get_dummies()` | Converted categorical → numeric |
| **7️⃣ Model Training** | `DecisionTreeClassifier().fit()` | Trained model on clean data |
| **8️⃣ Evaluation** | `accuracy_score()` | Measured predictive accuracy |
| **9️⃣ Prediction** | `to_csv('submission.csv')` | Produced final Kaggle output |

*(You can embed code snippets or screenshots here for clarity.)*

---

## 📈 Results  
| Model | Accuracy | Notes |
|--------|-----------|-------|
| Decision Tree | ~0.82 | Interpretable baseline |
| Random Forest | ~0.85 | Higher generalization |
| XGBoost | ~0.86 | Best overall result |

📊 *Feature importance showed that `Sex`, `Pclass`, and `Fare` were top predictors of survival.*

---

## 💡 Insights  
- Females and 1st-class passengers had significantly higher survival rates.  
- Smaller families had better survival odds compared to larger ones.  
- Ticket price (`Fare`) correlated positively with survival likelihood.  

---

## 🧠 Conclusion  
This project demonstrates my ability to apply the full machine learning workflow — from **data cleaning and feature engineering** to **model training and evaluation**.  
Using Python, pandas, seaborn, and scikit-learn, I built a predictive model to estimate passenger survival on the Titanic.  
It strengthened my skills in **data preprocessing, visualization, model evaluation**, and reinforced the value of data-driven problem solving.  

---

## 🛠️ Tools & Libraries  
- Python 3.x  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  
- Jupyter Notebook  

---

## 🚀 Future Improvements  
- Tune hyperparameters using GridSearchCV  
- Add ensemble models (e.g., Voting Classifier)  
- Deploy as a simple web app using **Streamlit**  

---

## 📎 Files in Repository  
| File | Description |
|------|--------------|
| `Titanic Challenges.ipynb` | Main analysis & model notebook |
| `train.csv`, `test.csv` | Dataset files |
| `submission.csv` | Final predictions for Kaggle |

---

### 🧑‍💻 Author  
**Eshita Adhikary**  
Data Science & Machine Learning Enthusiast  
📫 [eshita.adhikary91@example.com] | 🌐 [(https://www.linkedin.com/feed/)]

