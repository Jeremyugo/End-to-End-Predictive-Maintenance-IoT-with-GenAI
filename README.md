# End-to-End-Predictive-Maintenance-ETL-with-GenAI

An end-to-end project for real-time sensor data processing, fault detection, predictive maintenance, and GenAI-assisted repair operations.

## 🚀 Project Overview

This project implements a scalable data and AI pipeline using Spark Delta Lake, MLFlow, Dagster and LangChain to monitor wind turbines, detect faults, and support maintenance teams through GenAI agents. It includes:

- **Streaming ETL** with Delta Live Tables
- **Fault prediction** using a RandomForest ML model
- **Workflow orchestration** using Dagster
- **GenAI agent deployment** to assist turbine maintenance

---

## 🔧 Components

### 1. Data Ingestion
- Real-time sensor data streamed into Delta Lake via **Spark Delta Lake**
- Data quality checks and schema enforcement

### 3. Analytics & Dashboards
- Aggregated turbine metrics and fault trends
- Built using **Power BI**

### 4. Machine Learning
- Model to detect faulty turbines
- Triggers alerts for predictive maintenance actions
- Trained and deployed via **MLflow**

### 5. Workflow Orchestration
- **Dagster** used to schedule and automate pipeline steps

### 6. GenAI Agent System
- Custom GenAI agents assist technicians:
  - Diagnostic recommendations
  - Repair procedures
  - Interactive troubleshooting


### TODO:
- [x] Create Data Ingestion and Transformation scripts
- [x] Create Model Training Script
- [x] Create Model Evaluation Script
- [x] Create Model Registration Script
- [x] Create Dagster Pipeline with asset definitions
- [x] Build AI-Agent
- [ ] Build Streamlit FrontEnd
- [ ] Create MS Power BI dashboard
- [ ] Implement Data Drift Check