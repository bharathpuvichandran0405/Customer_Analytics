import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ─── 0) Let the user upload their own CSV ─────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Upload your cleaned CSV here (will override sample)", 
    type="csv",
    help="If you have the full df_cleaned.csv, upload it to see the complete data."
)

@st.cache_data
def load_sample():
    url = (
        "https://raw.githubusercontent.com/"
        "bharathpuvichandran0405/Customer_Analytics/main/df_cleaned_sample.csv"
    )
    return pd.read_csv(url)

if uploaded is not None:
    # read the uploaded file in memory
    df = pd.read_csv(uploaded)
else:
    df = load_sample()

# ─── rest of your app unchanged ───────────────────────────────────────────────
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.sidebar.title("📊 Customer Analytics Project")
page = st.sidebar.radio("Go to", [
    "Customer Lifetime Value",
    "Churn Prediction",
    "Product Recommendation"
])

if page == "Customer Lifetime Value":
    st.title("📈 Customer Lifetime Value Prediction")
    if 'Segment' in df.columns:
        order = ['High Value Customers','Medium Value Customers','Low Value Customers']
        counts = df['Segment'].value_counts().reindex(order)
        fig, ax = plt.subplots(figsize=(7,5))
        counts.plot(kind='bar', edgecolor='black', ax=ax)
        ax.set_title("Customer Count per CLV Segment")
        ax.set_xlabel("Segment"); ax.set_ylabel("Number of Customers")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
    else:
        st.warning("No 'Segment' column found in your data.")

elif page == "Churn Prediction":
    st.title("⚠️ Churn Prediction")
    if 'Predicted Churn' in df.columns:
        churn_counts = df['Predicted Churn'].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.pie(churn_counts, labels=['Active (0)','Churned (1)'],
                autopct='%1.1f%%', startangle=140)
        ax1.set_title("Churn Distribution (Pie)")
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        churn_counts.plot(kind='bar', color=['green','red'], edgecolor='black', ax=ax2)
        ax2.set_title("Churn Distribution (Bar)")
        ax2.set_xlabel("Churn Label"); ax2.set_ylabel("Number of Customers")
        st.pyplot(fig2)
    else:
        st.warning("No 'Predicted Churn' column found in your data.")

else:  # Product Recommendation
    st.title("🎯 Product Recommendation System")
    if 'Customer ID' not in df.columns:
        st.error("No 'Customer ID' column found in your data.")
    else:
        custs = sorted(df['Customer ID'].unique())
        sel = st.selectbox("Select a Customer ID", custs)
        if st.button("Get Recommendations"):
            # your real recommend logic here
            placeholder = [
                "WHITE BEADED GARLAND STRING 20LIGHT",
                "EASTER DECORATION NATURAL CHICK",
                "SKY BLUE COLOUR GLASS GEMS IN BAG",
                "CLASSIC DIAMANTE EARRINGS JET",
                "BAKING MOULD TOFFEE CUP CHOCOLATE"
            ]
            st.success(f"Top 5 recommendations for Customer {sel}:")
            for item in placeholder:
                st.write("✔️", item)
