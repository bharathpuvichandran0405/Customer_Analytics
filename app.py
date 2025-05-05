# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

# ─── 1) File upload ─────────────────────────────────────────────────────────
st.sidebar.header("🔄 Upload your Excel")
uploaded_file = st.sidebar.file_uploader(
    "Choose the `online_retail_II.xlsx` file", type="xlsx"
)
if uploaded_file is None:
    st.sidebar.info("Please upload your dataset to proceed.")
    st.stop()

# read into pandas
df = pd.read_excel(uploaded_file, sheet_name=0)

# ─── 2) Data cleaning & RFM feature engineering ────────────────────────────
df_cleaned = df.dropna(subset=["Customer ID"]).copy()
df_cleaned["Description"] = df_cleaned["Description"].fillna("Unknown")
df_cleaned["Customer ID"] = df_cleaned["Customer ID"].astype(int)
df_cleaned = df_cleaned[df_cleaned["Quantity"] > 0]

snapshot_date = df_cleaned["InvoiceDate"].max() + dt.timedelta(days=1)
rfm = (
    df_cleaned
    .groupby("Customer ID")
    .agg(
        Recency=lambda x: (snapshot_date - x.max()).days,
        Frequency=("Invoice", "count"),
        Monetary=lambda x: (x * df_cleaned.loc[x.index, "Quantity"]).sum()
    )
)

# score into quartiles
rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4]).astype(int)
rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)

def seg(score):
    if score>=10: return "High Value Customers"
    if score>=6:  return "Medium Value Customers"
    return "Low Value Customers"

rfm["Segment"] = rfm["RFM_Score"].apply(seg)

# churn label
rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

# ─── 3) Streamlit layout ────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Analytics", layout="wide")
st.title("📊 Customer Analytics Dashboard")

page = st.sidebar.radio("Go to", [
    "Customer Lifetime Value",
    "Churn Prediction",
    "Product Recommendation"
])

# ─── 4) Customer Lifetime Value page ────────────────────────────────────────
if page == "Customer Lifetime Value":
    st.header("📈 CLV Segments")
    order = ["High Value Customers","Medium Value Customers","Low Value Customers"]
    counts = rfm["Segment"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Segment")
    st.pyplot(fig)

# ─── 5) Churn Prediction page ────────────────────────────────────────────────
elif page == "Churn Prediction":
    st.header("⚠️ Churn Rates by Segment")
    # split features
    X = rfm[["Recency","Frequency","Monetary"]]
    y = rfm["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = (preds==y_test).mean()
    st.write(f"**Test accuracy:** {acc:.2%}")

    # overall churn pie
    churn_counts = rfm["Churn"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        churn_counts,
        labels=["Active","Churned"],
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title("Predicted Churn Distribution")
    st.pyplot(fig)

# ─── 6) Product Recommendation page ─────────────────────────────────────────
else:
    st.header("🎯 Product Recommendations")
    # build item similarity once
    matrix = df_cleaned.pivot_table(
        index="Customer ID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(matrix.T)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

    cust = st.selectbox("Choose Customer ID", rfm.index.tolist())
    st.write("You selected:", cust)

    if st.button("Get top‑5 recommendations"):
        user = matrix.loc[cust]
        bought = user[user>0].index.tolist()
        scores = pd.Series(dtype=float)
        for item in bought:
            scores = scores.add(sim_df[item], fill_value=0)
        scores = scores.drop(bought).sort_values(ascending=False)
        for rec in scores.head(5).index:
            st.write("✔️", rec)
