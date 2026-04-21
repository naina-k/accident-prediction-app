import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Accident AI Dashboard", layout="wide")

st.title("🚦 Accident Analytics & ML Training Dashboard")

data = pd.read_csv("dataset.csv")
model = pickle.load(open("model.pkl", "rb"))

# Sidebar
st.sidebar.header("Dataset Overview")
st.sidebar.write("Total Records:", data.shape[0])

# Dataset Preview
st.subheader("📊 Accident Dataset Preview")
st.dataframe(data.head(20))

# Training Section
st.subheader("🤖 Model Training & Testing")
data = pd.read_csv("dataset.csv")

# ✅ FIX HERE: Convert Weather text → numbers
data = pd.read_csv("dataset.csv")

# ✅ Encode weather
weather_map = {
    "Clear": 0,
    "Rain": 1,
    "Fog": 2
}
data["Weather"] = data["Weather"].map(weather_map)

# ✅ Load model
model = pickle.load(open("model.pkl", "rb"))


# ✅ DEFINE X and y (THIS WAS MISSING ❌)
X = data.drop("Severity", axis=1)
y = data["Severity"]

# ✅ Prediction
predictions = model.predict(X)

data["Weather"] = data["Weather"].map(weather_map)

predictions = model.predict(X)
accuracy = (predictions == y).mean()

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots()
ax.matshow(cm)
st.pyplot(fig)

# Analytics
st.subheader("📈 Accident Severity Analysis")
st.bar_chart(data["Severity"].value_counts())
st.subheader("🔮 Predict Accident Severity")

speed = st.number_input("Speed")
visibility = st.number_input("Visibility")
wind = st.number_input("Wind Speed")
hour = st.number_input("Hour")

weather_input = st.selectbox("Weather", ["Clear", "Rain", "Fog"])

weather_map = {"Clear":0, "Rain":1, "Fog":2}
weather = weather_map[weather_input]

if st.button("Predict"):
    result = model.predict([[speed, visibility, wind, hour, weather]])
    st.success(f"Predicted Severity: {result[0]}")