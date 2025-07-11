from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd

app = Flask(__name__)

# ----------------------------------------------------------
# ✅ Load model and feature order
# ----------------------------------------------------------
MODEL_PATH = 'depression_model.pkl'
FEATURE_PATH = 'feature_order.json'

print("[✅] Loading model...")
pipeline = joblib.load(MODEL_PATH)

print("[✅] Loading feature order...")
with open(FEATURE_PATH) as f:
    feature_order = json.load(f)

print("[✅] Model and feature order loaded!")

# ----------------------------------------------------------
# ✅ Health check
# ----------------------------------------------------------
@app.route('/')
def health():
    return "Mental Health Model is running!", 200

# ----------------------------------------------------------
# ✅ Prediction endpoint
# ----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data['features']
        
        # Wrap in DataFrame with correct columns
        input_df = pd.DataFrame([user_input], columns=feature_order)
        
        # Get prediction
        prediction = pipeline.predict(input_df)[0]
        
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ----------------------------------------------------------
# ✅ Run
# ----------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
