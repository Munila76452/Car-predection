from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

app = Flask(__name__)

# Load and prepare the model
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    return None

def prepare_model():
    # Load the data
    df = pd.read_csv('quikr_car.csv')
    
    # Clean the data
    df = df[df['Price'] != 'Ask For Price']
    df['Price'] = df['Price'].astype(float)
    df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '').astype(float)
    
    # Prepare features
    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    # Save the model and encoder
    with open('model.pkl', 'wb') as file:
        pickle.dump((model, encoder), file)
    
    return model, encoder

# Initialize model
model, encoder = prepare_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        data = request.get_json()
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'name': [data['name']],
            'company': [data['company']],
            'year': [int(data['year'])],
            'kms_driven': [float(data['kms_driven'])],
            'fuel_type': [data['fuel_type']]
        })
        
        # Transform the input data
        input_encoded = encoder.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 