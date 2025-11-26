
# EMI Analyzer Project

**EMI_Analyzer_Project** is a comprehensive financial risk assessment and EMI prediction platform. It integrates data preprocessing, machine learning model development, and a Streamlit-based interactive dashboard to provide actionable insights on loan risk and EMI calculations.

---

## 🏗️ Project Overview

The project is designed to analyze financial datasets, predict Equated Monthly Installments (EMIs), and assess credit risk. It leverages machine learning models for classification and regression tasks, tracks experiments using MLflow, and presents results through an interactive web interface using Streamlit.

Key objectives:

* Automate EMI prediction based on customer financial data.
* Evaluate risk associated with loan applicants.
* Track and manage ML experiments efficiently.
* Provide an easy-to-use web interface for end-users.

---

## 🗂️ Project Structure

```
EMI_Analyzer_Project/
│
├── data/                      # Dataset files
│   ├── emi_prediction_dataset.csv
│   ├── X_processed.csv
│   └── cleaned_data.csv
│
├── models/                    # Trained machine learning models and feature pipelines
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   └── feature_pipeline.pkl
│
├── notebooks/                 # Jupyter notebooks for experiments and exploratory data analysis (EDA)
│
├── src/                       # Source code
│   ├── mlflow_artifacts/      # MLflow experiment artifacts
│   ├── mlruns/                # MLflow run tracking
│   └── ...                    # Additional source code modules
│
├── streamlit_app/             # Streamlit web application
│   └── app.py                 # Main Streamlit app file
│
├── requirements.txt           # Python dependencies
│
└── README.md                  # Project documentation
```

---

## ⚙️ Features

1. **Data Preprocessing**

   * Cleans and transforms raw EMI datasets.
   * Generates processed datasets ready for model training.

2. **Machine Learning Models**

   * **Classification model:** Assesses financial risk of applicants.
   * **Regression model:** Predicts EMI amounts accurately.
   * Feature pipelines for consistent preprocessing during training and prediction.

3. **Experiment Tracking**

   * MLflow integration to log experiments, metrics, parameters, and artifacts.

4. **Interactive Dashboard**

   * Built with Streamlit.
   * Allows users to input financial data and receive predictions.
   * Visualizes risk scores, EMI values, and model performance metrics.

5. **Modular Design**

   * Clear separation of data, models, notebooks, and web application for maintainability.

---

## 💻 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mansidavane21/EMI_Analyzer_Project.git
cd EMI_Analyzer_Project
```

### 2️⃣ Setup Virtual Environment (Recommended)

```bash
python -m venv .venv
```

* **Windows:**

```bash
.venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

* After running, Streamlit will provide a local URL (usually `http://localhost:8501`) to open the interactive dashboard in a browser.
* Input customer financial details to predict EMI and assess risk.

---

## 📊 Usage

1. Explore datasets in the `data/` folder.
2. Perform experiments and EDA in `notebooks/`.
3. Train models using the provided pipelines.
4. Launch the Streamlit app for real-time predictions and insights.
5. MLflow logs can be explored in the `mlruns/` directory for tracking experiments.

---

## 🔧 Contributing

We welcome contributions! Steps to contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push to your branch:

```bash
git push origin feature-name
```

5. Open a Pull Request on GitHub.

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


## 📌 Future Improvements

* Add automated hyperparameter tuning for models.
* Implement advanced visualizations in the Streamlit dashboard.
* Integrate with a database for persistent storage of predictions.
* Deploy the app to a cloud platform for public access.



## 📁 Acknowledgements

* [Streamlit](https://streamlit.io/) for interactive web applications.
* [MLflow](https://mlflow.org/) for experiment tracking.
* Open-source Python libraries: pandas, scikit-learn, matplotlib, seaborn, etc.

