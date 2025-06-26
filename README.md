# End-to-End-Predictive-Maintenance-ETL-with-GenAI

A comprehensive end-to-end solution for real-time wind turbine monitoring, fault detection, and AI-assisted maintenance operations. This project combines modern data engineering practices, machine learning, and generative AI to create a robust predictive maintenance system.

![Gen-AI](./images/GenAI.png)

## 🚀 Project Overview

This project implements a scalable data and AI pipeline using cutting-edge technologies to monitor wind turbines, detect potential faults, and provide intelligent maintenance support. The system processes real-time sensor data, predicts potential failures, and offers AI-powered maintenance assistance.

### Key Features

- **Real-time Data Processing**: Streaming sensor data ingestion using Spark Structured Streaming
- **Scalable Storage**: Delta Lake integration for reliable data storage and versioning
- **Automated ML Pipeline**: Continuous model training, hyperparameter tuning, data drift detection, and evaluation with Sklearn, HyperOpt, Evidently and MLflow
- **Intelligent Maintenance**: GenAI-powered maintenance assistant using LangChain and OpenAI
- **Modern Frontend**: Interactive Streamlit-based user interface
- **Robust Orchestration**: Automated workflows with Dagster

## 🏗️ Architecture

### 1. Data Layer
- **Data Ingestion**
  - Spark Structured Streaming for real-time data processing
  - Support for multiple data formats (JSON, Parquet)
  - Automated schema inference and validation
  - Delta Lake integration with checkpoint management
  - Data quality checks and enforcement

### 2. Machine Learning Pipeline
- **Data Preparation**
  - Automated feature engineering
  - Custom sklearn transformers for datetime handling
  - Robust data preprocessing pipeline

- **Model Training**
  - RandomForest Classifier for fault prediction
  - Hyperparameter optimization using hyperopt
  - MLflow experiment tracking and model versioning
  - Automated data drift detection using Evidently AI

### 3. GenAI Assistant
- **Interactive Agent**
  - Built with LangChain and OpenAI GPT-3.5
  - Maintenance prediction capabilities
  - Turbine specifications retrieval
  - Contextual conversation memory
  - Natural language interface for technicians

### 4. Workflow Orchestration
- **Dagster Pipeline**
  - Separate data ingestion and model training pipelines
  - Continuous data ingestion using sensors
  - Scheduled model training (every 3 hours)
  - Automated asset dependencies management
  - Comprehensive logging and monitoring

### 5. User Interface
- **Streamlit Frontend**
  - Interactive chat interface with AI agent
  - Real-time query processing
  - Persistent conversation history
  - User-friendly session management
  - Responsive design

## 🛠️ Technology Stack

- **Data Processing**: Apache Spark, Delta Lake
- **Machine Learning**: scikit-learn, MLflow
- **AI/LLM**: LangChain, OpenAI GPT-3.5
- **Orchestration**: Dagster
- **Frontend**: Streamlit
- **Monitoring**: Evidently AI
- **Visualization**: Power BI (planned)

## 📊 Data Flow

1. Real-time sensor data streams into the system
2. Data is processed and stored in Delta Lake tables
3. ML pipeline continuously trains and evaluates models
4. GenAI agent processes maintenance queries
5. Results are presented through the Streamlit interface

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Apache Spark
- OpenAI API key
- MLflow server
- Dagster instance


## 🔍 Project Status

### Completed
- ✅ Data Ingestion and Transformation pipeline
- ✅ Machine Learning model development
- ✅ Model evaluation and registration system
- ✅ Dagster pipeline implementation
- ✅ AI Agent development
- ✅ Streamlit frontend
- ✅ Data drift monitoring

### In Progress
- 🔄 Power BI dashboard development
- 🔄 Additional model optimizations
- 🔄 Enhanced AI agent capabilities