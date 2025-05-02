import streamlit as st
import pandas as pd
import pickle

with open("drop_predictor.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍎 Produce Drop and Return Predictor (Batch Mode)")

produce_types = st.multiselect(
    "Select one or more Produce Types",
    ["Apple", "Onion", "Potato", "Stonefruit", "Bread", "Banana"]
)

total_people = st.number_input("Enter total number of people", min_value=0, step=1)

if st.button("Predict Drops and Returns"):
    if total_people == 0:
        for produce in produce_types:
            st.success(f"📦 {produce}: Drops = 0, Returns = 0")
    elif not produce_types:
        st.warning("Please select at least one produce type.")
    else:
        input_df = pd.DataFrame({
            "Produce Type": produce_types,
            "total": [total_people] * len(produce_types)
        })

        predictions = model.predict(input_df)

        results_df = pd.DataFrame(predictions, columns=["Estimated Drops", "Estimated Returns"])
        results_df.insert(0, "Produce Type", produce_types)
        st.subheader("📊 Prediction Results")
        st.dataframe(results_df.style.format({
            "Estimated Drops": "{:.0f}",
            "Estimated Returns": "{:.0f}"
        }))