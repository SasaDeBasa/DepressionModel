from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained pipeline
model = joblib.load('depression_model.pkl')

@app.route('/')
def health():
    return "✅ Depression Model is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        prediction = model.predict([features])
        result = int(prediction[0])
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
