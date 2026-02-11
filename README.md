# 📱 Mobile Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

The **Mobile Price Prediction App** is a Machine Learning–based web application that predicts the **price category of a mobile phone** based on its hardware specifications.  
The application classifies mobiles into four categories:

- 📉 Low Cost  
- 💰 Medium Cost  
- 💎 High Cost  
- 👑 Very High Cost  

The system is built using **Python, Scikit-Learn, and Streamlit**, providing an interactive interface for real-time predictions along with **probability confidence visualization**.

---

## 🎯 Problem Statement

Given various mobile specifications such as battery power, RAM, camera quality, screen size, and processor details, the goal is to **predict the mobile phone price range** accurately using Machine Learning classification algorithms.

---

## 📊 Dataset Description

- **Dataset Name:** Mobile_data.csv  
- **Source:** Public ML classification dataset  
- **Target Variable:** `Price` (0–3)

### 📁 Features in the Dataset

| Feature Name | Description |
|-------------|-------------|
| battery_power | Battery capacity (mAh) |
| clock_speed | Processor speed (GHz) |
| fc | Front camera megapixels |
| int_memory | Internal storage (GB) |
| m_dep | Mobile thickness (cm) |
| mobile_wt | Mobile weight (grams) |
| n_cores | Number of CPU cores |
| pc | Primary camera megapixels |
| px_height | Pixel resolution height |
| px_width | Pixel resolution width |
| ram | RAM size (MB) |
| sc_h | Screen height (cm) |
| sc_w | Screen width (cm) |
| talk_time | Talk time (hours) |
| Price | Price category (0–3) |

---

## 🧠 Machine Learning Models Used

- ✅ **Logistic Regression**
- 🌳 **Decision Tree Classifier**
- 🌲 **Random Forest Classifier**

Users can dynamically switch between models from the UI.

---

## 📈 Key Features

- Interactive Streamlit Web App  
- Real-time mobile price prediction  
- Multiple ML model selection  
- Probability confidence bar chart  
- Clean and beginner-friendly UI  
- Academic & industry project ready  

---

## 🛠️ Tech Stack Used

| Category | Technology |
|--------|-----------|
| Programming Language | Python |
| Frontend | Streamlit |
| ML Library | Scikit-Learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| IDE | VS Code / PyCharm |

---

## 🚀 Steps to Run the Project

1. Clone the repository and move into the project folder:
```bash
git clone https://github.com/your-username/mobile-price-prediction.git
cd mobile-price-prediction
Install the required libraries:

pip install streamlit pandas numpy scikit-learn matplotlib
Run the Streamlit application:

streamlit run app.py


---

## 📊 Output

- Predicted mobile price category  
- Confidence score visualization using probability chart  
- User-selected machine learning model results  


## 🎓 Academic Use

This project is suitable for:
- B.E Mini Project  
- B.E Major Project  
- Machine Learning Lab  
- Streamlit Dashboard Demonstration  

It clearly demonstrates the **complete machine learning workflow** from  
**data preprocessing → model training → prediction → visualization**.


## 🔧 Customization

You can modify:
- Machine learning model selection  
- Feature inputs from the sidebar  
- Dataset used for training and testing  
- Visualization components (charts, probabilities)  


## 🚀 Future Enhancements

- Model accuracy comparison  
- Feature importance visualization  
- Confusion matrix and performance metrics  
- Online deployment using Streamlit Cloud  
- Authentication-based dashboard  


## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project for educational purposes.


## 👩‍💻 Author

**Geethanjali M**  
B.E Student | Machine Learning Enthusiast  

