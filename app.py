"""
Simple Streamlit app - loads the trained perceptron from model.json,
takes two binary inputs from the user and shows the prediction.
"""

import json
import numpy as np
import streamlit as st

st.title("Binary Pattern Classifier")

# load the model that train.py saved
with open("model.json") as f:
    model = json.load(f)

w = np.array(model["weights"])
b = model["bias"]
gate = model["gate"]

st.write(f"This perceptron was trained on the **{gate}** gate.")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# two dropdowns so the user can only pick 0 or 1
x1 = st.selectbox("Input 1", [0, 1])
x2 = st.selectbox("Input 2", [0, 1])

if st.button("Predict"):
    z = w[0] * x1 + w[1] * x2 + b   # weighted sum
    prob = sigmoid(z)               # probability between 0 and 1
    label = 1 if prob >= 0.5 else 0  # anything above 0.5 counts as class 1

    st.subheader(f"Predicted class: {label}")
    st.write(f"Sigmoid probability: {prob:.4f}")

st.divider()

# show what the model actually learned
st.subheader("Learned parameters")
st.write(f"Weight 1 (w1): {w[0]:.4f}")
st.write(f"Weight 2 (w2): {w[1]:.4f}")
st.write(f"Bias (b): {b:.4f}")
st.caption(f"Formula used:  sigmoid({w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {b:.2f})")
