from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load model
model = joblib.load('depression_model.pkl')

# Load feature order
with open("feature_order.json") as f:
    feature_order = json.load(f)

@app.route('/')
def health():
    return "✅ Depression Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']

        if len(features) != len(feature_order):
            return jsonify({'error': f'Expected {len(feature_order)} features but got {len(features)}'}), 400

        input_df = pd.DataFrame([features], columns=feature_order)
        prediction = model.predict(input_df)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
