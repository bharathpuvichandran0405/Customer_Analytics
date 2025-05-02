import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1) Load your cleaned sample data from GitHub raw ─────────────────────────
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/"
        "bharathpuvichandran0405/Customer_Analytics/main/df_cleaned_sample.csv"
    )
    return pd.read_csv(url)

df = load_data()

# ─── 2) Streamlit page setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.sidebar.title("📊 Customer Analytics Project")
page = st.sidebar.radio(
    "Go to",
    ["Customer Lifetime Value", "Churn Prediction", "Product Recommendation"]
)

# ─── 3) Customer Lifetime Value ───────────────────────────────────────────────
if page == "Customer Lifetime Value":
    st.title("📈 Customer Lifetime Value Prediction")
    if "Segment" in df.columns:
        seg_order = ["High Value Customers","Medium Value Customers","Low Value Customers"]
        segment_counts = df["Segment"].value_counts().reindex(seg_order)
        fig, ax = plt.subplots(figsize=(7,5))
        segment_counts.plot(kind="bar", edgecolor="black", ax=ax)
        ax.set_title("Customer Count per CLV Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Number of Customers")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)
    else:
        st.warning("No 'Segment' column found in your data.")

# ─── 4) Churn Prediction ───────────────────────────────────────────────────────
elif page == "Churn Prediction":
    st.title("⚠️ Churn Prediction")
    if "Predicted Churn" in df.columns:
        churn_counts = df["Predicted Churn"].value_counts().sort_index()
        # Pie
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.pie(churn_counts, labels=["Active (0)","Churned (1)"], autopct="%1.1f%%", startangle=140)
        ax1.set_title("Churn Prediction Distribution (Pie)")
        st.pyplot(fig1)
        # Bar
        fig2, ax2 = plt.subplots(figsize=(6,4))
        churn_counts.plot(kind="bar", color=["green","red"], edgecolor="black", ax=ax2)
        ax2.set_title("Churn Prediction Distribution (Bar)")
        ax2.set_xlabel("Churn Label")
        ax2.set_ylabel("Number of Customers")
        st.pyplot(fig2)
    else:
        st.warning("No 'Predicted Churn' column found in your data.")

# ─── 5) Product Recommendation ─────────────────────────────────────────────────
else:
    st.title("🎯 Product Recommendation System")
    if "Customer ID" not in df.columns:
        st.error("No 'Customer ID' column found in your data.")
    else:
        customer_ids = sorted(df["Customer ID"].unique())
        selected_id = st.selectbox("Select a Customer ID", customer_ids)
        if st.button("Get Recommendations"):
            # Insert your real recommendation logic here
            recs = [
                "WHITE BEADED GARLAND STRING 20LIGHT",
                "EASTER DECORATION NATURAL CHICK",
                "SKY BLUE COLOUR GLASS GEMS IN BAG",
                "CLASSIC DIAMANTE EARRINGS JET",
                "BAKING MOULD TOFFEE CUP CHOCOLATE",
            ]
            st.success(f"Top 5 recommendations for Customer {selected_id}:")
            for r in recs:
                st.write("✔️", r)
