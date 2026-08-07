import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(x,y)

st.title("Simple Iris Species Predictor")

sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal length (cm)")
petal_width = st.number_input("Petal Width (cm)")

predict = st.button("Predict Species")

if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    species = iris.target_names[pred]
    st.success(f"Predicted Species: **{species}**")

st.markdown("---")
st.write("Adjust the input values and click Predict")