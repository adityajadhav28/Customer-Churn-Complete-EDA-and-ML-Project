#  Customer Churn Prediction & Interactive Analytics Dashboard

An end-to-end **Data Analytics**, **Machine Learning**, and **Streamlit** project that analyzes customer behavior, predicts customer churn, and provides an interactive web application for real-time predictions and business insights.

The project combines **data analysis**, **machine learning**, and **web deployment** to help telecom companies identify customers who are at risk of leaving and make data-driven retention decisions.

---

##  Live Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers directly impacts revenue, making churn prediction a critical business problem.

This project follows a complete Data Science workflow—from raw data to an interactive prediction application.

### The project includes:

*  Data Cleaning & Preprocessing
*  Exploratory Data Analysis (EDA)
*  Business Insight Generation
*  Feature Engineering
*  Machine Learning Model Development
*  Model Comparison & Evaluation
*  Feature Importance Analysis
*  Model Serialization using Joblib
*  Interactive Streamlit Web Application

---

#  Project Objectives

The primary objectives of this project are:

* Analyze customer behavior using Exploratory Data Analysis.
* Discover the major factors responsible for customer churn.
* Build predictive machine learning models.
* Compare multiple algorithms and select the best-performing model.
* Generate actionable business insights.
* Deploy the trained model through an interactive Streamlit application for real-time predictions.

---

#  Dataset

The dataset contains customer information such as:

* Customer Demographics
* Customer Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Internet Service
* Payment Method
* Online Security
* Tech Support
* Streaming Services
* Multiple Lines
* Churn Status (Target Variable)

---

# 🛠️ Technologies Used

| Category                | Technologies          |
| ----------------------- | --------------------- |
| Programming Language    | Python                |
| Data Manipulation       | Pandas, NumPy         |
| Data Visualization      | Matplotlib, Seaborn   |
| Machine Learning        | Scikit-learn, XGBoost |
| Model Saving            | Joblib                |
| Web Application         | Streamlit             |
| Development Environment | Jupyter Notebook      |

---

#  Exploratory Data Analysis (EDA)

The EDA focuses on answering important business questions, including:

* What percentage of customers churn?
* Which contract type has the highest churn rate?
* Does customer tenure influence churn?
* How do monthly charges affect churn?
* Which additional services improve customer retention?
* Which customer segments are most likely to leave?

Various visualizations were created to uncover hidden trends and support business decision-making.

---

#  Machine Learning Models

Multiple classification algorithms were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

The best-performing model was selected based on multiple evaluation metrics.

---

#  Model Evaluation Metrics

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

#  Key Business Insights

The analysis revealed several valuable insights:

* Customers with **Month-to-Month contracts** are significantly more likely to churn.
* Customers with **short tenure** have the highest probability of leaving.
* **Higher monthly charges** are associated with increased churn rates.
* Customers subscribed to **Online Security** and **Tech Support** services show better retention.
* Long-term contracts greatly improve customer loyalty.

These insights can help businesses design effective customer retention strategies and reduce revenue loss.

---

# 🌐 Streamlit Web Application

An interactive Streamlit application has been developed to make the machine learning model accessible through an intuitive web interface.

### Features

* User-friendly interface
* Real-time churn prediction
* Dynamic input fields
* Instant prediction results
* Probability-based risk estimation
* Professional dashboard layout
* Responsive design

The application allows users to enter customer information and instantly determine whether the customer is likely to churn.

---

#  Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_EDA_and_Machine_Learning.ipynb
├── app.py
├── dataset.csv
├── model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
│
├── images/
│   ├── churn_distribution.png
│   ├── contract_vs_churn.png
│   ├── feature_importance.png
│   └── streamlit_dashboard.png
│
└── assets/
```

---

#  Project Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance
      │
      ▼
Save Best Model
      │
      ▼
Deploy with Streamlit
```


---

#  Running the Project

### Clone the repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
```

### Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🧠 Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Data Visualization
* Machine Learning
* Classification Algorithms
* Model Evaluation
* Predictive Analytics
* Business Insight Generation
* Streamlit Application Development
* Model Deployment Preparation
* Data Storytelling

---

# 🚀 Future Enhancements

* Hyperparameter Tuning
* Cross Validation
* SHAP Explainability
* Customer Churn Probability Score
* Power BI Executive Dashboard
* Flask/FastAPI REST API Deployment
* Docker Containerization
* Cloud Deployment (AWS / Azure / Render)

---

# 📬 Contact

**Aditya Jadhav**

📧 Email: adityajadhav232812@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/aditya-jadhav-4a1376258/

💻 GitHub: https://github.com/adityajadhav28

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star** on GitHub.

Feedback and contributions are always welcome!

