from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("tyresinkage.pkl", "rb"))
scaler_X = pickle.load(open("X_scaledTS.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        normal_load = float(data.get("normal_load", 0))
        inflation_pressure = float(data.get("inflation_pressure", 0))
        soil = data.get("soil_condition", "").lower()
        DP = float(data.get("DP", 0))
        slip = float(data.get("slip", 0))

        sc = 1 if soil == "soft" else 0

        features = np.array([[normal_load, inflation_pressure, sc, DP, slip]])
        scaled_features = scaler_X.transform(features)
        prediction = model.predict(scaled_features)[0]

        return jsonify({"tyre_sinkage_mm": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)