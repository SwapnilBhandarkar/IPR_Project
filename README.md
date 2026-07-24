# 📈 AI-Powered Time Series Forecasting Dashboard

> A production-inspired machine learning dashboard that combines **Classical Statistics**, **Machine Learning**, **Deep Learning**, and **Large Language Models (LLMs)** into a unified forecasting platform.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-success)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project is an **interactive Time Series Forecasting Dashboard** developed during my academic internship at the **Institute for Plasma Research (IPR), Gandhinagar**.

The application enables users to upload time-series datasets, compare multiple forecasting algorithms, evaluate model performance using standard statistical metrics, visualize predictions interactively, and automatically generate natural language insights using **Meta LLaMA 3.3** served through the **Groq API**.

Unlike traditional forecasting tools that only generate predictions, this platform bridges the gap between **predictive analytics** and **explainable AI** by converting numerical evaluation metrics into easy-to-understand analytical summaries.

---





```
                    Dataset
                       │
                       ▼
              Data Preprocessing
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
 Classical        Machine Learning   Deep Learning
 ARIMA            Random Forest      LSTM
 SARIMA           Prophet
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             Model Evaluation
        Accuracy • MAE • MSE • RMSE
                       │
                       ▼
          Groq API (LLaMA 3.3-70B)
                       │
                       ▼
      AI Generated Forecast Interpretation
```

---

# Key Features

### Multi-Model Forecasting

Implemented and compared five forecasting paradigms:

- ARIMA
- SARIMA
- Prophet
- Random Forest
- LSTM

Each model follows its own optimized preprocessing and training pipeline while sharing a common evaluation framework.

---

### Interactive Dashboard

Developed using **Streamlit** with a responsive interface.

Features include:

- Interactive time-series visualization
- Model selection
- Forecast horizon customization
- One-click training
- Future forecasting
- Residual analysis
- Cross-model comparison
- AI-generated insights

---

### Automated Evaluation

Every model is evaluated using:

- Accuracy (%)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The dashboard enables objective comparison across different forecasting techniques.

---

### AI-Powered Interpretation

One of the core contributions of this project is the integration of **Groq API** with **Meta LLaMA 3.3 (70B)**.

Instead of only displaying metrics, the system automatically generates:

- performance summaries
- trend interpretation
- model suitability
- forecast explanation

making the dashboard accessible even for non-technical users.

---



# Machine Learning Pipeline

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Chronological Train/Test Split
   │
   ▼
Model Training
   │
   ▼
Prediction
   │
   ▼
Performance Metrics
   │
   ▼
Visualization
   │
   ▼
LLM-Based Insight Generation
```

---



# Model Comparison

| Model | Type |
|---------|----------------------------|
| ARIMA | Statistical Forecasting |
| SARIMA | Seasonal Statistical Forecasting |
| Prophet | Additive Time Series Model |
| Random Forest | Machine Learning |
| LSTM | Deep Learning |

The dashboard allows side-by-side comparison of all models under identical preprocessing conditions.


---

# Highlights

- End-to-end forecasting platform
- Five forecasting algorithms
- Interactive visualization
- Automated model comparison
- LLM-assisted interpretation
- Modular architecture
- Production-inspired workflow
- Explainable AI integration
- Clean and extensible codebase

---

# Future Enhancements

- Transformer-based forecasting (PatchTST, TFT)
- AutoML model selection
- Bayesian Hyperparameter Optimization
- Real-time streaming using Kafka/MQTT
- Cloud Deployment (AWS/GCP/Azure)
- Multi-user authentication


---

---
# Installation

```bash
git clone https://github.com/SwapnilBhandarkar/Time-Series-Forecasting.git

cd Time-Series-Forecasting

pip install -r requirements.txt

streamlit run app.py
```

---

# Skills Demonstrated

- Time Series Forecasting
- Machine Learning
- Deep Learning
- Statistical Modeling
- Python Development
- Feature Engineering
- Data Visualization
- Model Evaluation
- Explainable AI
- LLM Integration
- Streamlit Development
- Software Architecture
- Predictive Analytics

---

# About

This project was developed as an Academic Research Project at the **Institute for Plasma Research (IPR), Gandhinagar**, focusing on building a unified forecasting platform capable of integrating traditional forecasting methods, machine learning, deep learning, and large language models into a single interactive analytics dashboard.
