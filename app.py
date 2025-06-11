import streamlit as st
import pandas as pd

st.set_page_config(page_title="Transaction Anomaly Detection System", layout="wide")

st.title("Transaction Anomaly Detection System")
st.write("This tool allows fraud investigators to assess whether a transaction is potentially anomalous based on key indicators.")

# ===== REQUIRED FIELDS =====
st.header("Required Transaction Information")
actual_price = st.number_input("Original Price (USD)", min_value=0.0)
discounted_price = st.number_input("Discounted Price (USD)", min_value=0.0)

# Auto calculate discount percentage
discount_percentage = 0.0
if actual_price > 0:
    discount_percentage = round((actual_price - discounted_price) / actual_price * 100, 2)
    st.markdown(f"**Calculated Discount Percentage:** {discount_percentage}%")
else:
    st.warning("Please enter an original price greater than 0 to calculate discount percentage.")

rating = st.selectbox("Product Rating (stars)", options=[1, 2, 3, 4, 5])
rating_count = st.number_input("Number of Ratings", min_value=0)

# ===== OPTIONAL FIELDS =====
st.header("Optional Additional Information")
category = st.text_input("Product Category")
about_product = st.text_area("Product Description")
review_title = st.text_input("Review Title")
review_content = st.text_area("Review Content")
product_link = st.text_input("Product Link")

# ===== ANOMALY LOGIC =====
def classify_transaction(actual_price, discounted_price, discount_percentage, rating, rating_count):
    reasons = []
    if discount_percentage > 50 and rating <= 2:
        reasons.append("Extremely high discount with low rating")
    if rating_count <= 2 and discounted_price < 10:
        reasons.append("Very few ratings and price is too low")
    if actual_price >= 500 and discounted_price <= 20:
        reasons.append("High original price but extremely low final price")
    return (1 if reasons else 0), reasons

if st.button("Analyze Transaction"):
    result, reasons = classify_transaction(actual_price, discounted_price, discount_percentage, rating, rating_count)
    if result == 1:
        st.error("⚠️ Potential Fraudulent Transaction Detected")
        st.write("**Reasons:**")
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.success("✅ Transaction Appears Normal")
        st.markdown("No suspicious indicators found.")
