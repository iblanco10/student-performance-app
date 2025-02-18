from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os  # Import os here, at the top

# Load the trained model and label encoder from the models_data directory
model = joblib.load("models_data/student_performance_model.pkl")
label_encoder = joblib.load("models_data/label_encoder.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        math_score = float(request.form['math_score'])
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])

        # Prepare data for prediction
        input_data = np.array([[math_score, reading_score, writing_score]])
        prediction = model.predict(input_data)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'predicted_race_ethnicity': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Moved here
    app.run(debug=True, host="0.0.0.0", port=port)  # Keep this under the if __name__ block
