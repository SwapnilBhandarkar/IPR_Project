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

# Project Architecture

<p align="center">
<img src="images/architecture.png" width="900">
</p>

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

- Dataset preview
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

# Tech Stack

| Category | Technologies |
|------------|----------------------------|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Statistical Models | Statsmodels |
| Machine Learning | Scikit-Learn |
| Deep Learning | TensorFlow, Keras |
| Forecasting | Prophet |
| Visualization | Plotly |
| LLM Integration | Groq API |
| AI Model | Meta LLaMA 3.3 70B |
| Environment | Python 3.10 |

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

# Dashboard Modules

- Data Overview
- Forecast Visualization
- Residual Analysis
- Model Comparison
- Performance Metrics
- AI Insights Panel

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

# Screenshots

## Dashboard

```
images/dashboard.png
```

<img src="images/dashboard.png">

---

## Forecast Visualization

```
images/forecast.png
```

<img src="images/forecast.png">

---

## Model Comparison

```
images/comparison.png
```

<img src="images/comparison.png">

---

## Residual Analysis

```
images/residual.png
```

<img src="images/residual.png">

---

## AI Insights

```
images/ai_insights.png
```

<img src="images/ai_insights.png">

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
- REST API support
- Docker containerization

---

# Repository Structure

```
Time-Series-Forecasting/
│
├── app.py
├── dataset/
├── models/
│   ├── arima.py
│   ├── sarima.py
│   ├── prophet.py
│   ├── random_forest.py
│   └── lstm.py
│
├── utils/
├── images/
├── notebooks/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

```bash
git clone https://github.com/yourusername/Time-Series-Forecasting.git

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
