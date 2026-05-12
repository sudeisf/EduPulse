
---

# EduPulse Intelligence System

**Project Title:** EduPulse – Big Data Student Performance Intelligence System  
**Author:** [Your Name]  
**Category:** Big Data / Machine Learning / Data Engineering  

## Project Overview
EduPulse is a distributed data pipeline and predictive engine designed to analyze massive volumes of academic and behavioral data (10M+ rows). By utilizing **Apache Spark**, the system performs complex joins and feature engineering to identify hidden patterns that correlate with academic success. 

The system features two Machine Learning engines:
1. **Classification (Risk Model):** Predicts the probability of a student failing or dropping out.
2. **Regression (GPA Model):** Predicts a student's final numerical grade based on current behavior.

---

## 📂 ZIP Folder Structure
Ensure the folder is organized as follows before submitting:
```text
EduPulse/
├── data/               
│   └── raw/             <-- TEACHER: Place OULAD CSV files here
├── src/                 <-- Modular source code (pipelines, modeling, core)
├── notebooks/           <-- (Optional) Any EDA notebooks
├── Dockerfile           <-- Environment definition
├── docker-compose.yml   <-- Orchestration
├── requirements.txt     <-- Dependencies
└── README.md            <-- This guide
```

---

## 🛠️ Setup Instructions

### Option A: The "One-Click" Docker Method (Recommended)
This method is preferred as it handles all dependencies (Java, Spark, Python 3.9) automatically.

1. **Place Data:** Extract the OULAD CSV files into `data/raw/`.
2. **Build & Start:** Open a terminal in the project root and run:
   ```bash
   docker-compose up --build -d
   ```
3. **Execute Pipeline:** Run the intelligence scripts in order:
   ```bash
   docker-compose exec edupulse-app python src/pipelines/ingestion.py
   docker-compose exec edupulse-app python src/pipelines/engagement.py
   docker-compose exec edupulse-app python src/modeling/risk_model.py
   docker-compose exec edupulse-app python src/modeling/regression_model.py
   docker-compose exec edupulse-app python src/pipelines/analytics.py
   ```
4. **View Dashboard:** Navigate to `http://localhost:8501`.

---

### Option B: Local Execution (Without Docker)
Use this if Docker is not available. **Requirements:** Java 8/11 and Python 3.9-3.11.

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install setuptools    # Required for Python 3.12 compatibility
   ```
2. **Place Data:** Ensure CSVs are in `data/raw/`.
3. **Run Pipeline (Order is Critical):**
   ```bash
   python src/pipelines/ingestion.py
   python src/pipelines/engagement.py
   python src/modeling/risk_model.py
   python src/modeling/regression_model.py
   python src/pipelines/analytics.py
   ```
4. **Start UI:**
   ```bash
   streamlit run src/app.py
   ```

---

## Technical Highlights for Evaluators

### 1. Data Scale & Optimization
The system processes the **OULAD dataset**, specifically the `studentVle.csv` which contains over **10 million rows**. To handle this on local hardware, the project implements:
*   **Columnar Storage:** Conversion of raw CSVs to **Apache Parquet**, reducing file size by ~80% and increasing read speeds.
*   **Lazy Evaluation:** Leveraging Spark’s DAG to optimize transformations before execution.

### 2. Feature Engineering (Engagement Index)
The project creates a custom **Behavioral Engagement Index**. It doesn't just count clicks; it aggregates time-series interactions into a weighted score of "Days Active" vs. "Interaction Intensity," providing a superior input for the ML models.

### 3. Dual-Model Architecture
*   **Random Forest Classifier:** Trained to identify the binary risk of failure with high recall.
*   **Linear Regression:** Designed to provide a "Speedometer" view of a student's predicted final mark, allowing for more granular academic tracking.

### 4. Interactive Analytics
The **Institutional Analytics** tab performs distributed aggregations to show regional and demographic performance gaps, demonstrating the use of Spark for high-level BI (Business Intelligence).

---

##  Conclusion
EduPulse demonstrates a full-stack Big Data lifecycle: from raw ingestion and distributed cleaning to machine learning and interactive visualization. It moves educational data from **descriptive** (what happened) to **prescriptive** (what will happen).

**Note to Evaluator:** Please ensure you have at least 4GB of RAM allocated to Docker/Spark for the best experience when processing the 10M row interaction logs.