# End-to-End-Predictive-Maintenance-ETL-with-GenAI

An end-to-end Databricks project for real-time sensor data processing, fault detection, predictive maintenance, and GenAI-assisted repair operations.

## 🚀 Project Overview

This project implements a scalable data and AI pipeline using Databricks to monitor wind turbines, detect faults, and support maintenance teams through GenAI agents. It includes:

- **Streaming ETL** with Delta Live Tables
- **Data governance** via Unity Catalog
- **Data visualization** with Databricks SQL and Warehouse
- **Fault prediction** using ML models
- **Workflow orchestration** using Databricks Workflows
- **GenAI agent deployment** to assist turbine maintenance

---

## 🔧 Components

### 1. Data Ingestion
- Real-time sensor data streamed into Delta Lake via **Delta Live Tables**
- Data quality checks and schema enforcement

### 2. Data Governance
- Assets managed and secured using **Unity Catalog**
- Role-based access controls for teams

### 3. Analytics & Dashboards
- Aggregated turbine metrics and fault trends
- Built using **Databricks SQL + Warehouse Endpoints**

### 4. Machine Learning
- Model to detect faulty turbines
- Triggers alerts for predictive maintenance actions
- Trained and deployed via **Databricks MLflow**

### 5. Workflow Orchestration
- **Databricks Workflows** used to schedule and automate pipeline steps

### 6. GenAI Agent System
- Custom GenAI agents assist technicians:
  - Diagnostic recommendations
  - Repair procedures
  - Interactive troubleshooting



