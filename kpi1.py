"""Streamlit dashboard for Olist analytics and late-delivery prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from train_model import MODEL_FEATURES, prepare_order_data


st.set_page_config(page_title="Olist Dashboard", page_icon="🛒", layout="wide")
st.title("🛒 Olist E-Commerce Dashboard")
st.write("Business analytics and late-delivery prediction")


@st.cache_data(show_spinner="Loading and cleaning the Olist datasets...")
def load_data() -> pd.DataFrame:
    return prepare_order_data()


@st.cache_resource(show_spinner="Loading the pre-trained model...")
def load_model():
    """Load the model created earlier by train_model.py; never train here."""
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "model.pkl is missing. Run train_model.py before launching Streamlit."
        )
    return load(model_path)


df = load_data()
st.success(f"Dataset loaded successfully: {len(df):,} delivered orders")

page = st.sidebar.selectbox("Select Page", ["Dashboard", "ML Prediction"])
st.sidebar.header("Filters")

selected_state = st.sidebar.selectbox(
    "Customer State", ["All"] + sorted(df["customer_state"].unique().tolist())
)
state_df = df if selected_state == "All" else df[df["customer_state"] == selected_state]

selected_category = st.sidebar.selectbox(
    "Product Category",
    ["All"] + sorted(state_df["product_category"].unique().tolist()),
)
filtered_df = (
    state_df
    if selected_category == "All"
    else state_df[state_df["product_category"] == selected_category]
)

if filtered_df.empty:
    st.warning("No orders match the selected filters.")
    st.stop()


if page == "Dashboard":
    with st.expander("Dataset Preview"):
        st.dataframe(filtered_df.head(100), width="stretch")

    total_revenue = filtered_df["total_sales"].sum()
    total_freight = filtered_df["total_freight"].sum()
    total_orders = filtered_df["order_id"].nunique()
    average_order_value = total_revenue / total_orders
    average_delivery = filtered_df["delivery_days"].mean()
    late_rate = filtered_df["is_late"].mean()
    average_review = filtered_df["average_review_score"].mean()
    freight_share = total_freight / (total_revenue + total_freight)

    st.header("📈 KPI Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Revenue", f"R$ {total_revenue:,.2f}")
    col2.metric("📦 Orders", f"{total_orders:,}")
    col3.metric("🧾 Avg Order Value", f"R$ {average_order_value:,.2f}")
    col4.metric("🚚 Freight", f"R$ {total_freight:,.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("⏱ Avg Delivery", f"{average_delivery:.2f} days")
    col6.metric("⚠️ Late Rate", f"{late_rate:.2%}")
    col7.metric("⭐ Avg Review", f"{average_review:.2f} / 5")
    col8.metric("📊 Freight Share", f"{freight_share:.2%}")

    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("Revenue by Category")
        category_revenue = (
            filtered_df.groupby("product_category")["total_sales"]
            .sum()
            .nlargest(10)
        )
        st.bar_chart(category_revenue)

    with chart2:
        st.subheader("Late-Delivery Rate by State")
        state_delivery = filtered_df.groupby("customer_state").agg(
            orders=("order_id", "nunique"), late_rate=("is_late", "mean")
        )
        state_delivery = state_delivery[state_delivery["orders"] >= 50]
        st.bar_chart(state_delivery["late_rate"].nlargest(10))

    chart3, chart4 = st.columns(2)
    with chart3:
        st.subheader("Monthly Revenue Trend")
        monthly_revenue = filtered_df.groupby("purchase_period")["total_sales"].sum()
        st.line_chart(monthly_revenue)

    with chart4:
        st.subheader("Top Customer States by Revenue")
        state_revenue = (
            filtered_df.groupby("customer_state")["total_sales"]
            .sum()
            .nlargest(10)
        )
        st.bar_chart(state_revenue)

    st.header("💡 Business Recommendations")
    recommendations = []
    if late_rate > 0.10:
        recommendations.append("Audit carriers and sellers because the late rate is above 10%.")
    if freight_share > 0.15:
        recommendations.append("Review shipping contracts because freight exceeds 15% of order value.")
    if average_review < 4:
        recommendations.append("Prioritize service recovery because the average review is below 4/5.")

    top_category = category_revenue.index[0]
    recommendations.append(f"Protect inventory availability in {top_category}, the top revenue category.")

    if not state_delivery.empty:
        highest_risk_state = state_delivery["late_rate"].idxmax()
        recommendations.append(
            f"Investigate logistics in {highest_risk_state}, the highest late-rate state in this view."
        )

    for recommendation in recommendations:
        st.success(recommendation)

    st.header("📊 Automated Insights")
    top_state = state_revenue.index[0]
    st.write(f"🏆 **Top category:** {top_category}")
    st.write(f"🌎 **Top customer state:** {top_state}")
    st.write(f"📦 **Orders analyzed:** {total_orders:,}")

    review_by_delivery = filtered_df.groupby("delivery_status")[
        "average_review_score"
    ].mean()
    if {"On Time", "Late"}.issubset(review_by_delivery.index):
        review_gap = review_by_delivery["On Time"] - review_by_delivery["Late"]
        st.write(
            f"⭐ **Delivery effect:** on-time orders score {review_gap:.2f} review points higher than late orders."
        )

    st.header("📌 Business Status")
    if late_rate <= 0.10:
        st.success("Delivery performance is within the 10% late-rate guardrail.")
    else:
        st.warning("Delivery performance is outside the 10% late-rate guardrail.")

else:
    st.header("🤖 Late-Delivery Prediction")
    st.write("Enter information available when the order is placed.")

    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        sales_input = st.number_input(
            "Item Value (R$)", min_value=0.0, value=float(df["total_sales"].median())
        )
        freight_input = st.number_input(
            "Freight Value (R$)", min_value=0.0, value=float(df["total_freight"].median())
        )
        item_count_input = st.number_input("Number of Items", min_value=1, value=1)
        estimated_days_input = st.number_input(
            "Estimated Delivery Days",
            min_value=1.0,
            value=float(round(df["estimated_delivery_days"].median(), 1)),
        )

    with col2:
        month_input = st.slider("Purchase Month", 1, 12, 6)
        category_input = st.selectbox(
            "Product Category", sorted(df["product_category"].unique())
        )
        customer_state_input = st.selectbox(
            "Customer State", sorted(df["customer_state"].unique())
        )
        seller_state_input = st.selectbox(
            "Seller State", sorted(df["seller_state"].unique())
        )

    input_data = pd.DataFrame(
        {
            "total_sales": [sales_input],
            "total_freight": [freight_input],
            "item_count": [item_count_input],
            "estimated_delivery_days": [estimated_days_input],
            "purchase_month": [month_input],
            "product_category": [category_input],
            "customer_state": [customer_state_input],
            "seller_state": [seller_state_input],
        }
    )[MODEL_FEATURES]

    if st.button("Predict Delivery Risk", type="primary"):
        prediction = int(model.predict(input_data)[0])
        late_probability = float(model.predict_proba(input_data)[0, 1])
        if prediction == 1:
            st.error(f"Likely Late ⚠️ — probability: {late_probability:.2%}")
        else:
            st.success(f"Likely On Time ✅ — probability: {1 - late_probability:.2%}")
        st.write(f"Late probability: **{late_probability:.2%}**")
