"""Streamlit dashboard for Olist analytics and late-delivery prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from train_model import MODEL_FEATURES, prepare_order_data


# Configure the browser tab and introduce the two purposes of the application.
st.set_page_config(page_title="Olist Dashboard", page_icon="🛒", layout="wide")
st.title("🛒 Olist E-Commerce Dashboard")
st.write("Business analytics and late-delivery prediction")


# Cache the cleaned order-level dataset so filter changes do not repeat the ETL work.
@st.cache_data(show_spinner="Loading and cleaning the Olist datasets...")
def load_data() -> pd.DataFrame:
    """Load, clean, merge, and aggregate the Olist source tables."""
    return prepare_order_data()


# Cache the fitted pipeline and keep model training outside the Streamlit process.
@st.cache_resource(show_spinner="Loading the pre-trained model...")
def load_model():
    """Load the model created earlier by train_model.py; never train here."""
    model_path = Path(__file__).resolve().parent / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "model.pkl is missing. Run train_model.py before launching Streamlit."
        )
    return load(model_path)


# Load the shared analytical dataset once for the dashboard and prediction form.
df = load_data()
st.success(f"Dataset loaded successfully: {len(df):,} delivered orders")


# Build the sidebar navigation and apply state/category filters to dashboard results.
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

# Stop cleanly when a filter combination has no matching delivered orders.
if filtered_df.empty:
    st.warning("No orders match the selected filters.")
    st.stop()


if page == "Dashboard":
    # Let users inspect a small sample without overwhelming the main dashboard.
    with st.expander("Dataset Preview"):
        st.dataframe(filtered_df.head(100), width="stretch")

    # Calculate the headline commercial, delivery, review, and freight KPIs.
    total_revenue = filtered_df["total_sales"].sum()
    total_freight = filtered_df["total_freight"].sum()
    total_orders = filtered_df["order_id"].nunique()
    late_orders = int(filtered_df["is_late"].sum())
    average_order_value = total_revenue / total_orders
    average_delivery = filtered_df["delivery_days"].mean()
    late_rate = filtered_df["is_late"].mean()
    on_time_rate = 1 - late_rate
    average_review = filtered_df["average_review_score"].mean()
    freight_share = (
        total_freight / (total_revenue + total_freight)
        if total_revenue + total_freight > 0
        else 0.0
    )

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

    # Prepare reusable category, state, and monthly summaries for charts and insights.
    category_summary = (
        filtered_df.groupby("product_category", as_index=True)
        .agg(
            revenue=("total_sales", "sum"),
            freight=("total_freight", "sum"),
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )
    category_summary["freight_share"] = category_summary["freight"] / (
        category_summary["revenue"] + category_summary["freight"]
    ).replace(0, pd.NA)

    state_summary = (
        filtered_df.groupby("customer_state", as_index=True)
        .agg(
            revenue=("total_sales", "sum"),
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
            average_delivery=("delivery_days", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )
    monthly_revenue = (
        filtered_df.groupby("purchase_period")["total_sales"].sum().sort_index()
    )

    # Require a reasonable sample before identifying a state or category as high risk.
    minimum_group_orders = max(1, min(50, round(total_orders * 0.05)))
    eligible_states = state_summary[state_summary["orders"] >= minimum_group_orders]
    eligible_categories = category_summary[
        category_summary["orders"] >= minimum_group_orders
    ]

    # Display the four core business trends used by the automated analysis below.
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("Revenue by Category")
        st.bar_chart(category_summary["revenue"].head(10))

    with chart2:
        st.subheader("Late-Delivery Rate by State")
        st.bar_chart(eligible_states["late_rate"].nlargest(10))

    chart3, chart4 = st.columns(2)
    with chart3:
        st.subheader("Monthly Revenue Trend")
        st.line_chart(monthly_revenue)

    with chart4:
        st.subheader("Top Customer States by Revenue")
        st.bar_chart(state_summary["revenue"].head(10))

    # Derive extra measures that explain concentration, retention, and delivery impact.
    top_category = category_summary.index[0]
    top_state = state_summary.index[0]
    top_category_share = (
        category_summary.iloc[0]["revenue"] / total_revenue
        if total_revenue > 0
        else 0.0
    )
    top_state_share = (
        state_summary.iloc[0]["revenue"] / total_revenue
        if total_revenue > 0
        else 0.0
    )
    top_three_category_share = (
        category_summary["revenue"].head(3).sum() / total_revenue
        if total_revenue > 0
        else 0.0
    )

    peak_period = monthly_revenue.idxmax()
    peak_period_revenue = monthly_revenue.max()
    latest_growth = None
    if len(monthly_revenue) >= 2 and monthly_revenue.iloc[-2] > 0:
        latest_growth = monthly_revenue.iloc[-1] / monthly_revenue.iloc[-2] - 1

    customer_order_counts = filtered_df.groupby("customer_unique_id")[
        "order_id"
    ].nunique()
    repeat_customer_rate = (
        (customer_order_counts > 1).mean() if not customer_order_counts.empty else 0.0
    )

    late_delays = (
        filtered_df.loc[filtered_df["is_late"].eq(1), "delivery_days"]
        - filtered_df.loc[
            filtered_df["is_late"].eq(1), "estimated_delivery_days"
        ]
    )
    average_late_delay = late_delays.mean() if not late_delays.empty else 0.0

    highest_risk_state = (
        eligible_states["late_rate"].idxmax() if not eligible_states.empty else None
    )
    highest_risk_category = (
        eligible_categories["late_rate"].idxmax()
        if not eligible_categories.empty
        else None
    )
    highest_freight_category = (
        eligible_categories["freight_share"].dropna().idxmax()
        if not eligible_categories["freight_share"].dropna().empty
        else None
    )

    review_by_delivery = filtered_df.groupby("delivery_status")[
        "average_review_score"
    ].mean()
    review_gap = None
    if (
        {"On Time", "Late"}.issubset(review_by_delivery.index)
        and review_by_delivery[["On Time", "Late"]].notna().all()
    ):
        review_gap = review_by_delivery["On Time"] - review_by_delivery["Late"]

    # Generate a larger set of data-driven actions, not generic fixed advice.
    st.header("💡 Business Recommendations")
    recommendations: list[tuple[str, str, str]] = []

    if late_rate > 0.10:
        recommendations.append(
            (
                "warning",
                "High priority",
                f"Audit carrier and seller fulfillment because {late_rate:.1%} of orders are late, above the 10% guardrail.",
            )
        )
    else:
        recommendations.append(
            (
                "success",
                "Maintain",
                f"Keep the current delivery controls and monitor exceptions; the {late_rate:.1%} late rate is within the 10% guardrail.",
            )
        )

    if freight_share > 0.15:
        recommendations.append(
            (
                "warning",
                "Cost",
                f"Renegotiate shipping rates or revise free-shipping thresholds because freight represents {freight_share:.1%} of collected order value.",
            )
        )
    else:
        recommendations.append(
            (
                "info",
                "Cost",
                f"Continue monitoring freight by route; its current {freight_share:.1%} share is below the 15% alert level.",
            )
        )

    if pd.notna(average_review) and average_review < 4:
        recommendations.append(
            (
                "warning",
                "Customer experience",
                f"Prioritize service recovery and seller quality checks because the average review is {average_review:.2f}/5.",
            )
        )
    elif pd.notna(average_review):
        recommendations.append(
            (
                "success",
                "Customer experience",
                f"Preserve the practices supporting the {average_review:.2f}/5 review score and investigate low-score exceptions.",
            )
        )

    recommendations.append(
        (
            "info",
            "Inventory",
            f"Protect stock availability and seller coverage in {top_category}, which generates {top_category_share:.1%} of filtered revenue.",
        )
    )

    if highest_risk_state is not None:
        risky_state = eligible_states.loc[highest_risk_state]
        recommendations.append(
            (
                "warning",
                "Geography",
                f"Review routes, carriers, and promised dates in {highest_risk_state}; its late rate is {risky_state['late_rate']:.1%} across {int(risky_state['orders']):,} orders.",
            )
        )

    if highest_risk_category is not None:
        risky_category = eligible_categories.loc[highest_risk_category]
        recommendations.append(
            (
                "warning",
                "Seller operations",
                f"Inspect seller handling and packaging SLAs for {highest_risk_category}; it has the highest eligible category late rate at {risky_category['late_rate']:.1%}.",
            )
        )

    if highest_freight_category is not None:
        freight_category_share = eligible_categories.loc[
            highest_freight_category, "freight_share"
        ]
        recommendations.append(
            (
                "info",
                "Shipping efficiency",
                f"Test bundling, regional fulfillment, or minimum-order thresholds for {highest_freight_category}, where freight reaches {freight_category_share:.1%} of collected value.",
            )
        )

    if review_gap is not None and review_gap > 0.25:
        recommendations.append(
            (
                "warning",
                "Reviews",
                f"Send proactive delay notifications and recovery offers; late delivery is associated with a {review_gap:.2f}-point review penalty.",
            )
        )

    if average_late_delay > 3:
        recommendations.append(
            (
                "warning",
                "Escalation",
                f"Introduce carrier escalation before the promised date because late orders miss it by {average_late_delay:.1f} days on average.",
            )
        )

    if repeat_customer_rate < 0.10:
        recommendations.append(
            (
                "info",
                "Retention",
                f"Launch post-purchase and second-order campaigns; only {repeat_customer_rate:.1%} of customers in this view placed more than one order.",
            )
        )
    else:
        recommendations.append(
            (
                "success",
                "Retention",
                f"Build loyalty offers around the {repeat_customer_rate:.1%} repeat-customer base.",
            )
        )

    if latest_growth is not None:
        if latest_growth < 0:
            recommendations.append(
                (
                    "warning",
                    "Revenue trend",
                    f"Investigate availability, demand, and campaign changes because the latest monthly revenue fell {abs(latest_growth):.1%}.",
                )
            )
        else:
            recommendations.append(
                (
                    "success",
                    "Revenue trend",
                    f"Identify the categories and states behind the latest {latest_growth:.1%} monthly growth and repeat the strongest actions.",
                )
            )

    if top_three_category_share > 0.60:
        recommendations.append(
            (
                "info",
                "Portfolio risk",
                f"Reduce category concentration by developing adjacent categories; the top three currently contribute {top_three_category_share:.1%} of revenue.",
            )
        )

    for message_type, label, recommendation in recommendations:
        getattr(st, message_type)(f"**{label}:** {recommendation}")

    # Explain the filtered results with quantified observations and sample sizes.
    st.header("📊 Automated Insights")
    insights = [
        f"🏆 **Category leader:** {top_category} contributes {top_category_share:.1%} of revenue.",
        f"🌎 **Geographic leader:** {top_state} contributes {top_state_share:.1%} of revenue.",
        f"🧺 **Revenue concentration:** the top three categories account for {top_three_category_share:.1%} of revenue.",
        f"📅 **Peak sales period:** {peak_period} generated R$ {peak_period_revenue:,.2f}.",
        f"📦 **Delivery outcome:** {late_orders:,} of {total_orders:,} orders were late; {on_time_rate:.1%} arrived on time.",
        f"👥 **Customer retention:** {repeat_customer_rate:.1%} of {len(customer_order_counts):,} customers placed more than one order.",
        f"🚚 **Freight burden:** freight represents {freight_share:.1%} of the total amount collected from products and shipping.",
    ]

    if late_orders:
        insights.append(
            f"⏳ **Late-order severity:** late deliveries missed the estimate by {average_late_delay:.1f} days on average."
        )
    else:
        insights.append(
            "⏳ **Late-order severity:** no late orders appear in the selected view."
        )

    if latest_growth is not None:
        direction = "increased" if latest_growth >= 0 else "decreased"
        insights.append(
            f"📈 **Latest monthly movement:** revenue {direction} by {abs(latest_growth):.1%} versus the previous period."
        )

    if highest_risk_state is not None:
        risky_state = eligible_states.loc[highest_risk_state]
        insights.append(
            f"🗺️ **Highest-risk state:** {highest_risk_state} has a {risky_state['late_rate']:.1%} late rate across {int(risky_state['orders']):,} eligible orders."
        )

    if highest_risk_category is not None:
        risky_category = eligible_categories.loc[highest_risk_category]
        insights.append(
            f"🛍️ **Highest-risk category:** {highest_risk_category} has a {risky_category['late_rate']:.1%} late rate across {int(risky_category['orders']):,} eligible orders."
        )

    if review_gap is not None:
        comparison = "higher" if review_gap >= 0 else "lower"
        insights.append(
            f"⭐ **Delivery effect on reviews:** on-time orders score {abs(review_gap):.2f} points {comparison} than late orders on average."
        )

    if highest_freight_category is not None:
        freight_category_share = eligible_categories.loc[
            highest_freight_category, "freight_share"
        ]
        insights.append(
            f"💸 **Highest freight burden:** {highest_freight_category} spends {freight_category_share:.1%} of collected value on freight."
        )

    for insight in insights:
        st.markdown(f"- {insight}")

    # Summarize whether the most important operating guardrails are healthy.
    st.header("📌 Business Status")
    status1, status2, status3 = st.columns(3)
    with status1:
        if late_rate <= 0.10:
            st.success("Delivery is within the 10% late-rate guardrail.")
        else:
            st.warning("Delivery is outside the 10% late-rate guardrail.")
    with status2:
        if pd.notna(average_review) and average_review >= 4:
            st.success("Customer reviews meet the 4/5 target.")
        else:
            st.warning("Customer reviews are below the 4/5 target.")
    with status3:
        if freight_share <= 0.15:
            st.success("Freight is within the 15% cost guardrail.")
        else:
            st.warning("Freight is above the 15% cost guardrail.")

else:
    # Load the saved pipeline only when the user opens the prediction page.
    st.header("🤖 Late-Delivery Prediction")
    st.write("Enter information available when the order is placed.")
    model = load_model()

    # Collect the same numeric and categorical inputs used during model training.
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

    # Preserve the exact feature names and order expected by the fitted pipeline.
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

    # Run inference with model.pkl and present both the class and late probability.
    if st.button("Predict Delivery Risk", type="primary"):
        prediction = int(model.predict(input_data)[0])
        late_probability = float(model.predict_proba(input_data)[0, 1])
        if prediction == 1:
            st.error(f"Likely Late ⚠️ — probability: {late_probability:.2%}")
        else:
            st.success(f"Likely On Time ✅ — probability: {1 - late_probability:.2%}")
        st.write(f"Late probability: **{late_probability:.2%}**")
