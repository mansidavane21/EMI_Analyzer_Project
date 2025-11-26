
# Imports

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie
import plotly.express as px
import io

# Page Configuration
st.set_page_config(
    page_title="FinTech EMI Platform",
    page_icon="🏦",
    layout="wide"
)


# Function: Load Lottie Animation

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar Navigation

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "💳 EMI Eligibility",
    "📈 Max EMI Prediction",
    "📊 EDA Visualizer",
    "⚙️ Model Performance"
])

# Top Title Banner

st.markdown(
    """
    <div style='
        padding: 20px;
        background-color: #0A1A44; 
        color: white;               
        border-radius: 12px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    '>
        🏦 FinTech EMI Risk Assessment Platform
    </div>
    """,
    unsafe_allow_html=True
)

# 🏠 Home Page
def show_home_page():
    
    st.title("🏠 Welcome to Smart EMI Analyzer")

    st.write("""
    **AI-powered EMI eligibility & affordability assessment — secure, explainable, and production-ready.**

    Take the guesswork out of your financial planning. Smart EMI Analyzer helps you understand how much EMI you can truly afford, using intelligent machine-learning models and real financial indicators.
    """)

    st.markdown("---")

    st.subheader("🌟 What You Can Do Here")

    st.markdown("""
    ### **1. Check EMI Eligibility**
    Get accurate predictions of whether you qualify for an EMI based on:
    - Monthly income  
    - Fixed expenses  
    - Existing EMIs  
    - Financial behavior  

    ---

    ### **2. Know Your Maximum Affordable EMI**
    Our AI model computes the safest EMI you can comfortably pay each month without creating financial stress.

    ---

    ### **3. Explore Your Data with EDA Visualizer**
    Upload your dataset and instantly access:
    - Summary statistics  
    - Financial trends  
    - Interactive charts  
    - Insights into customer behavior  

    ---

    ### **4. Understand Model Performance**
    Analyze your model with:
    - Accuracy metrics  
    - Error analysis  
    - Prediction quality  
    - Actual vs Predicted comparisons  
    
    """
               
                )

    st.markdown("---")

    st.subheader("🔐 Secure & Explainable")
    st.write("""
    Your data stays safe.  
    Every prediction comes with transparency and explainability so you always understand **why** the model made a decision.
    """)

    st.markdown("---")


    st.subheader("🚀 Built for Real-World Use")
    st.write("""
    Smart EMI Analyzer is built to be:
    - Fast  
    - Scalable  
    - User-friendly  
    - Business-ready  
    """)
    st.markdown("---")

    st.subheader("💡 Make Smarter Financial Decisions")
    st.write("""
    Whether you're planning a loan, improving creditworthiness, or analyzing customer financial patterns —  
    this platform gives you the clarity and confidence you need.
    """)
    st.markdown("---")


#EMI Eligibility  page

def show_emi_page():
    st.title("💳 Check Your EMI Eligibility")
    st.write("Upload your financial details or enter manually to check EMI eligibility and maximum affordable EMI.")

    # Manual Input
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Monthly Income (₹)", min_value=0, step=1000)
        existing_emi = st.number_input("Existing EMIs (₹)", min_value=0, step=500)
    with col2:
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=5000)
        tenure = st.slider("Loan Tenure (Months)", min_value=6, max_value=60, value=12)

    # EMI calculation function
    def predict_emi_eligibility(income, existing_emi, loan_amount, tenure):
        emi = (loan_amount / tenure) * 1.1
        max_emi_allowed = 0.5 * (income - existing_emi)
        status = "Eligible" if emi <= max_emi_allowed else "Not Eligible"
        emi_ratio = min(int((emi / max_emi_allowed) * 100), 100) if max_emi_allowed > 0 else 0
        return {"emi": round(emi,2), "max_emi": round(max_emi_allowed,2), "status": status, "emi_ratio": emi_ratio}

    # Button click
    if st.button("Check Eligibility"):
        result = predict_emi_eligibility(income, existing_emi, loan_amount, tenure)

        # Result
        st.subheader("✅ Result")
        st.markdown(
            f"**Status:** {result['status']}  \n"
            f"**Calculated EMI:** ₹{result['emi']}  \n"
            f"**Maximum Affordable EMI:** ₹{result['max_emi']}"
        )

        # Progress Bar
        st.subheader("EMI Affordability")
        st.progress(result['emi_ratio'])

        # Tips
        st.subheader("💡 Tips / Advisory")
        if result['status'] == "Eligible":
            st.success("You are eligible for this loan! Maintain your EMI obligations wisely.")
        else:
            st.error("You are not eligible. Consider reducing existing EMIs or requesting a lower loan amount.")

        # Pie chart
        st.subheader("📊 Loan & EMI Distribution")
        remaining_income = max(income - existing_emi - result['emi'], 0)
        values = [existing_emi, result['emi'], remaining_income]
        labels = ['Existing EMIs', 'Calculated EMI', 'Remaining Income']
        colors = ['#FF6B6B', '#4ECDC4', '#1A535C']

        fig, ax = plt.subplots(figsize=(3,2))
        ax.pie(values, labels=labels, autopct=lambda pct: f"₹{int(pct/100*sum(values))}", 
               startangle=120, colors=colors, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize':5})
        ax.axis('equal')
        ax.set_title(f"Eligibility Status: {result['status']}", fontsize=12, fontweight='bold')
        st.pyplot(fig)

        # Download report
        report_df = pd.DataFrame({
            "Monthly Income": [income],
            "Existing EMIs": [existing_emi],
            "Requested Loan": [loan_amount],
            "Tenure (Months)": [tenure],
            "Calculated EMI": [result['emi']],
            "Max Affordable EMI": [result['max_emi']],
            "Eligibility Status": [result['status']]
        })
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Report as CSV", data=csv, file_name="EMI_Eligibility_Report.csv")


# Max Emi prediction page

def show_max_emi_page():
    st.title("📈 Check Max EMI Prediction")
    st.write("Estimate the maximum EMI you can afford based on your income and current EMIs.")

    # Session state defaults
    for key, default in {"income": 0, "existing_emi": 0, "tenure": 12, "max_emi": 0}.items():
        if key not in st.session_state:
            st.session_state[key] = default

   
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Monthly Income (₹)", min_value=0, step=1000, key="income")
        existing_emi = st.number_input("Existing EMIs (₹)", min_value=0, step=500, key="existing_emi")
    with col2:
        tenure = st.slider("Preferred Loan Tenure (Months)", min_value=6, max_value=60, value=st.session_state["tenure"], key="tenure")

    # Max EMI Calculation
    def calculate_max_emi(income, existing_emi):
        disposable_income = income - existing_emi
        if disposable_income <= 0:
            return 0
        return round(0.5 * disposable_income, 2)

    
    # Predict Button
    
    if st.button("Predict Max EMI"):
        max_emi = calculate_max_emi(income, existing_emi)
        st.session_state["max_emi"] = max_emi

        # Result
        st.subheader("✅ Maximum Affordable EMI")
        st.markdown(f"Your maximum affordable EMI is: **₹{max_emi}** per month.")

        # Progress Bar
        st.subheader("💹 EMI vs Disposable Income")
        disposable_income = income - existing_emi
        ratio = int((max_emi / disposable_income) * 100) if disposable_income > 0 else 0
        st.progress(ratio)
        st.caption(f"Max EMI is {ratio}% of your disposable income (Income - Existing EMIs)")

        
        # Pie Chart
        
        st.subheader("📊 Income Distribution")
        remaining_income = max(disposable_income - max_emi, 0)
        values = [existing_emi, max_emi, remaining_income]
        labels = ["Existing EMIs", "Max EMI", "Remaining Income"]
        colors = ["#FF6B6B", "#4ECDC4", "#1A535C"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            hoverinfo="label+value+percent"
        )])
        fig.update_layout(title=f"Eligibility Status: {'Eligible' if max_emi>0 else 'Not Eligible'}", title_x=0.5)
        pie_chart = st.plotly_chart(fig, use_container_width=True)

    
        
        # EMI Breakdown Table
        st.subheader("📋 EMI Breakdown Over Tenure")
        # Calculate cumulative EMI and remaining balance
        cumulative = [round(max_emi * (i + 1)) for i in range(tenure)]
        remaining_balance = [round(max(max_emi * tenure - cumulative[i], 0)) for i in range(tenure)]

        #Create DataFrame
        emi_breakdown = pd.DataFrame({
        "Month": list(range(1, tenure + 1)),
        "EMI Amount (₹)": [round(max_emi)] * tenure,
        "Cumulative Paid (₹)": cumulative,
        "Remaining Balance (₹)": remaining_balance
        })

        # Display in Streamlit (scrollable and sortable)
        st.dataframe(emi_breakdown, use_container_width=True)

       
        # Download Report
        report_df = pd.DataFrame({
            "Monthly Income": [income],
            "Existing EMIs": [existing_emi],
            "Preferred Tenure (Months)": [tenure],
            "Max Affordable EMI": [max_emi]
        })
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Max EMI Report as CSV", data=csv, file_name="Max_EMI_Report.csv")




#EDA Visualizer page
def show_eda_page():
    st.title("📊 EDA Visualizer")
    st.write("Upload your financial dataset and explore it interactively with charts and statistics.")


    # File Upload
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("File loaded successfully! ✅")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        st.markdown("---")

        
        # Basic Info
        st.subheader("ℹ️ Dataset Overview")
        st.write("Shape of the dataset:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Missing values per column:")
        st.write(df.isnull().sum())

        st.markdown("---")

        
        # Select Column for Analysis
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        if numeric_cols:
            col_to_analyze = st.selectbox("Select a numeric column to visualize", numeric_cols)

            st.subheader(f"🔹 Distribution of {col_to_analyze}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df[col_to_analyze], kde=True, bins=20, color="#4ECDC4", ax=ax)
            ax.set_xlabel(col_to_analyze)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.subheader(f"🔹 Boxplot of {col_to_analyze}")
            fig2, ax2 = plt.subplots(figsize=(6,2))
            sns.boxplot(x=df[col_to_analyze], color="#FF6B6B", ax=ax2)
            st.pyplot(fig2)
        
        
        # Correlation Heatmap
        if len(numeric_cols) > 1:
            st.subheader("🔹 Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig3, ax3 = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

        
        # Pairplot (Optional)
        if st.checkbox("Show Pairplot (for numeric columns)"):
            st.write("Generating pairplot, please wait...")
            sns_plot = sns.pairplot(df[numeric_cols])
            st.pyplot(sns_plot)
        
       


# Model Performance page

def show_model_performance_page():

    st.title("📈 Model Performance Dashboard")
    st.write("Upload your model output containing Actual and Predicted EMI values.")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        # Load file
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except:
            st.error("❌ Unable to read the file.")
            return

        
        # Normalize all column names

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        
        # Accepted variations

        actual_variants = [
            "actual", "actual_emi", "y_test", "y_true", "target", "emi_actual"
        ]

        predicted_variants = [
            "predicted", "predicted_emi", "y_pred", "prediction",
            "model_output", "emi_predicted"
        ]

        
        # Auto-detect column names
        actual_col = None
        pred_col = None

        # detect actual column
        for col in df.columns:
            if col in actual_variants:
                actual_col = col
                break

        # detect predicted column
        for col in df.columns:
            if col in predicted_variants:
                pred_col = col
                break

       
        #  If not detected, show error
        if actual_col is None or pred_col is None:
            st.error("❌ Required Actual and Predicted columns not found.")
            st.write("Detected columns:", df.columns.tolist())
            st.info("""
            Make sure your file has columns such as:
            • Actual or actual_emi or y_test  
            • Predicted or predicted_emi or y_pred
            """)
            return

        # Assign detected columns
        actual = df[actual_col]
        predicted = df[pred_col]

        st.success(f"✅ Detected Actual Column → **{actual_col}**")
        st.success(f"✅ Detected Predicted Column → **{pred_col}**")

         
        # STEP 6 — Create error columns 
        df["error"] = actual - predicted
        df["absolute_error"] = abs(df["error"])

        
        #Metrics (use lowercase column names)
        mae = df["absolute_error"].mean()
        mse = (df["error"] ** 2).mean()
        rmse = np.sqrt(mse)

        # Avoid division by zero for MAPE
        if (actual == 0).any():
            mape = np.nan
        else:
            mape = (abs(df["error"] / actual).mean()) * 100

        # R2 Score
        try:
            r2 = 1 - (np.sum((actual - predicted) ** 2) /
                    np.sum((actual - np.mean(actual)) ** 2))
        except:
            r2 = np.nan
            
        # KPI Cards
        st.subheader("📊 Key Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("MAPE (%)", f"{mape:.2f}")
        col5.metric("R² Score", f"{r2:.3f}")

        st.write("---")

        
        # Actual vs Predicted Plot
        st.subheader("📌 Actual vs Predicted EMI")

        fig1 = plt.figure(figsize=(8, 4))
        plt.plot(actual.values, label="Actual EMI")
        plt.plot(predicted.values, label="Predicted EMI")
        plt.legend()
        plt.xlabel("Record Index")
        plt.ylabel("EMI Amount")
        plt.title("Actual vs Predicted EMI")
        st.pyplot(fig1)

        
        # Scatter Plot
        st.subheader("📌 Scatter Plot: Actual vs Predicted")

        fig2 = plt.figure(figsize=(6, 4))
        plt.scatter(actual, predicted)
        plt.xlabel("Actual EMI")
        plt.ylabel("Predicted EMI")
        plt.title("Scatter Plot: Model Fit")
        st.pyplot(fig2)

        
        # Residual Plot
        st.subheader("📌 Residual Distribution")

        fig3 = plt.figure(figsize=(6, 4))
        plt.hist(df["error"], bins=20)   # UPDATED LINE
        plt.xlabel("Error (Actual - Predicted)")
        plt.ylabel("Count")
        plt.title("Residual Distribution")
        st.pyplot(fig3)

        
        # Error Table
        st.subheader("📜 Error Table (High Risk Customers at Top)")
        st.dataframe(df.sort_values(by="absolute_error", ascending=False))

       
        # Risk Insights
        st.subheader("🔍 Risk Insights (Auto-Generated)")

        insights = []

        if mae > 500:
            insights.append("⚠️ High average error in EMI prediction — may affect loan risk evaluation.")

        if df["error"].mean() > 0:
            insights.append("🔺 Model **overestimates** EMI — may reject eligible customers.")

        if df["error"].mean() < 0:
            insights.append("🔻 Model **underestimates** EMI — may approve risky customers.")

        if mape > 20:
            insights.append("⚠️ Model is unstable with high percentage errors.")

        if len(insights) == 0:
            insights.append("✅ Model performance is stable and reliable.")

        for i in insights:
            st.write(i)

        st.write("---")

        # Download Report (PDF)
        
        st.subheader("📥 Download Model Performance Report")

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        st.download_button(
            label="Download CSV Report",
            data=buffer.getvalue(),
            file_name="model_performance_report.csv",
            mime="text/csv"
        )


# Page Routing

if page == "🏠 Home":
    show_home_page()
elif page == "💳 EMI Eligibility":
    show_emi_page()
elif page == "📈 Max EMI Prediction":
    show_max_emi_page()  
elif page == "📊 EDA Visualizer":
    show_eda_page()
elif page == "⚙️ Model Performance":
     show_model_performance_page()
else:
    st.warning("This page is under construction.")



