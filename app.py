from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("bangalore_price_model.pkl", "rb"))

# Load column data
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# -------------------- HOME PAGE --------------------
@app.route('/')
def home():
    # Send location list to dropdown (optional)
    locations = data_columns[3:]  # assuming first 3 are sqft, bath, bhk
    return render_template('index.html', locations=locations)


# -------------------- WEB FORM PREDICTION --------------------
@app.route('/predict_web', methods=['POST'])
def predict_web():
    try:
        sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        location = request.form['location']

        x = np.zeros(len(data_columns))

        x[data_columns.index('total_sqft')] = sqft
        x[data_columns.index('bath')] = bath
        x[data_columns.index('bhk')] = bhk

        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        prediction = model.predict([x])[0]

        return render_template(
            'index.html',
            prediction_text="Predicted Price: ₹ {:.2f} Lakhs".format(prediction),
            locations=data_columns[3:]
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="Error: {}".format(str(e)),
            locations=data_columns[3:]
        )


# -------------------- API PREDICTION (POSTMAN) --------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    sqft = float(data['total_sqft'])
    bath = int(data['bath'])
    bhk = int(data['bhk'])
    location = data['location']

    x = np.zeros(len(data_columns))

    x[data_columns.index('total_sqft')] = sqft
    x[data_columns.index('bath')] = bath
    x[data_columns.index('bhk')] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    prediction = model.predict([x])[0]

    return jsonify({
        "predicted_price": round(prediction, 2)
    })


# -------------------- RUN APP (RENDER READY) --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
