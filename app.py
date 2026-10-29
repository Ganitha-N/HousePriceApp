from flask import Flask, request, jsonify
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

df = pd.read_csv("train.csv")
df["bhk"] = pd.to_numeric(
    df["title"].str.extract(r"(\d+)")[0], errors="coerce"
)
df["property_type"] = df["title"].str.contains(
    "Villa", case=False, na=False
).map({True: "Villa", False: "Apartment"})

df = df.dropna(subset=[
    "area_insqft", "price(L)", "bhk", "location", "property_type"
])

X = df[["area_insqft", "bhk", "location", "property_type"]]
y = df["price(L)"]

model = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["location", "property_type"])
    ], remainder="passthrough")),
    ("reg", LinearRegression())
])
model.fit(X, y)

@app.route("/")
def home():
    return "House Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    house = pd.DataFrame([{
        "area_insqft": float(data["area"]),
        "bhk": float(data["bhk"]),
        "location": data["location"],
        "property_type": data["property_type"]
    }])
    price = float(model.predict(house)[0])
    return jsonify({"predicted_price_lakhs": round(price, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
