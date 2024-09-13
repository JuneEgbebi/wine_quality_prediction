from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')

@app.route('/')
def home():
    return render_template('index.html', model_columns=model_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [float(request.form[col]) for col in model_columns]
        
        # Convert to numpy array and reshape for scaler transformation
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features_array)
        
        # Make prediction using the model
        prediction = model.predict(features_scaled)
        
        # Return the prediction result to the HTML template
        return render_template('index.html', model_columns=model_columns, prediction_text=f'Predicted Quality: {prediction[0]}')
    
    except Exception as e:
        return render_template('index.html', model_columns=model_columns, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
