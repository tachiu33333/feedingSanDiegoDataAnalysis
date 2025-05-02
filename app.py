import streamlit as st
import pandas as pd
import pickle

with open("drop_predictor.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍎 Produce Drop Predictor")

produce_type = st.selectbox("Select Produce Type", [
    "Apple", "Onion", "Potato", "Stonefruit", "Bread", "Banana"
])

total_people = st.number_input("Enter total number of people", min_value=0, step=1)

if st.button("Predict Drops"):
    input_df = pd.DataFrame({
        "Produce Type": [produce_type],
        "total": [total_people]
    })
    predicted_drops = model.predict(input_df)[0]
    st.success(f"Estimated Drops: {int(predicted_drops)} items")
