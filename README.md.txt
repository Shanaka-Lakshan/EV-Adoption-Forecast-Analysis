# 🚗 EV Adoption Forecasting & Regional Analysis

## 📌 Overview

This project presents an end-to-end data analytics pipeline to **forecast and analyze electric vehicle (EV) adoption trends** for **regional infrastructure planning**. It combines data preprocessing, exploratory data analysis (EDA), machine learning models, clustering techniques, and an interactive dashboard.

The goal is to transform raw EV registration data into **actionable insights** for policymakers, planners, and researchers.

---

## 🎯 Key Features

* 📈 Forecast future EV adoption trends (up to 2030)
* 🗺️ Analyze regional adoption patterns across counties
* 🔍 Segment regions into adopter categories:

  * Early Adopters
  * Moderate Adopters
  * Slow Adopters
* ⚡ Evaluate charging infrastructure distribution
* 🖥️ Interactive dashboard for visualization and exploration

---

## 📊 Dataset

* ~279,000 EV records
* Covers 250+ counties and 40+ manufacturers
* Includes:

  * County, City
  * Model Year
  * Electric Range
  * Vehicle Type

📌 Publicly available dataset (Washington State EV data)

---

## ⚙️ Technologies Used

* **Python** (Pandas, NumPy)
* **Scikit-learn** (Machine Learning)
* **Matplotlib** (Visualization)
* **Streamlit** (Dashboard)

---

## 🧹 Data Preprocessing

* Removed records with missing critical attributes
* Filled missing Electric Range values
* Aggregated data by:

  * Year (for forecasting)
  * County (for regional analysis)
* Applied feature scaling for clustering

---

## 📈 Exploratory Data Analysis

Key observations:

* Rapid EV growth after 2018
* BEVs dominate over PHEVs
* Adoption concentrated in urban regions
* Charging infrastructure is unevenly distributed

---

## 🤖 Models Used

### **Linear Regression (Baseline)**

* Captures long-term adoption trend
* Used for forecasting

### **Random Forest Regression**

* Handles nonlinear relationships
* Improves pattern recognition

### **K-Means Clustering**

* Segments counties into adopter categories
* Based on EV count, range, and model year

---

## 📊 Key Insights

* EV adoption is accelerating rapidly
* Adoption is not uniform across regions
* Infrastructure availability strongly influences adoption
* Market is dominated by a few manufacturers

---

## 🖥️ Dashboard

### ▶️ Run Locally

```bash
streamlit run ev_dashboard.py
```

Then open:

```
http://localhost:8501
```

### 🌐 Access via GitHub (Streamlit Cloud)

To make the dashboard accessible online:

Go to: https://streamlit.io/cloud
Sign in with GitHub
Click **"New App"**
Select your repository
Set:
Branch: main
File: ev_dashboard.py
Click **Deploy**
👉 Your dashboard will be available via a public URL.

---

## 📂 Project Structure

```
EV_Project/
│
├── ev_dashboard.py
├── EV_Analysis.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   ├── ev_population.csv
│   ├── charging_stations.csv
│
└── outputs/
    ├── figures/

```

---

## 🔁 Reproducibility

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run analysis

* Open Jupyter Notebook:

```
EV_Analysis.ipynb
```

* Launch dashboard:

```bash
streamlit run ev_dashboard.py
```

---

##💻 System Requirements
Python 3.10+
RAM: 8 GB recommended
CPU: Multi-core processor
GPU: Not required

---

## ⚠️ Limitations

* Limited external factors (income, policy, fuel prices)
* Yearly aggregation reduces temporal detail
* Clustering assumes fixed number of groups (k=3)

---

## 🔮 Future Improvements

* Include socio-economic and policy variables
* Use advanced time-series models (Prophet, LSTM)
* Expand analysis to multiple regions

---

## 📌 Conclusion

This project demonstrates how machine learning and data analytics can support **evidence-based EV infrastructure planning**. By combining forecasting, clustering, and visualization, it provides actionable insights for policymakers and urban planners.
---

## 👨‍💻 Author

Shanaka Lakshan

