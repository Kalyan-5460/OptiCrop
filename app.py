import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: '{model_path}' not found! Please run train.py first.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('findyourcrop.html', 
                               prediction_text='Error: Prediction model is not loaded.')
    try:
        # Retrieve form values explicitly by name to ensure correctness
        nitrogen = float(request.form.get('nitrogen'))
        phosphorous = float(request.form.get('phosphorous'))
        potassium = float(request.form.get('potassium'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('ph'))
        rainfall = float(request.form.get('rainfall'))
        
        # Prepare feature array
        features = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        
        # Run prediction
        prediction = model.predict(features)
        output = prediction[0]
        
        # Capitalize output for presentation
        crop_recommended = output.capitalize()
        
        # Pass inputs back to preserve form state if needed, and the output
        return render_template('findyourcrop.html', 
                               prediction_text=f'Best crop for given conditions is {crop_recommended}',
                               prediction_success=True,
                               nitrogen=nitrogen,
                               phosphorous=phosphorous,
                               potassium=potassium,
                               temperature=temperature,
                               humidity=humidity,
                               ph=ph,
                               rainfall=rainfall)
    except Exception as e:
        return render_template('findyourcrop.html', 
                               prediction_text=f'Error processing input: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
