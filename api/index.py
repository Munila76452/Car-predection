from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import urllib.parse

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we need from app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Create a Flask app
app = Flask(__name__)

# Define a simple model for demonstration
def create_simple_model():
    # Create a simple dataset
    data = {
        'name': ['Honda City', 'Maruti Swift', 'Toyota Innova'],
        'company': ['Honda', 'Maruti', 'Toyota'],
        'year': [2015, 2016, 2017],
        'kms_driven': [45000, 35000, 55000],
        'fuel_type': ['Petrol', 'Petrol', 'Diesel'],
        'Price': [500000, 400000, 800000]
    }
    df = pd.DataFrame(data)
    
    # Prepare features
    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    
    # Train a simple model
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    return model, encoder

# Initialize model
model, encoder = create_simple_model()

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Car Price Predictor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .main-container {
                max-width: 800px;
                margin: 2rem auto;
                padding: 2rem;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            .form-label {
                font-weight: 500;
            }
            .result-container {
                display: none;
                margin-top: 2rem;
                padding: 1rem;
                border-radius: 5px;
                background-color: #e9ecef;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 1rem 0;
            }
        </style>
    </head>
    <body>
        <div class="container main-container">
            <h1 class="text-center mb-4">Car Price Predictor</h1>
            <form id="predictionForm">
                <div class="row g-3">
                    <div class="col-md-6">
                        <label for="name" class="form-label">Car Model</label>
                        <input type="text" class="form-control" id="name" required>
                    </div>
                    <div class="col-md-6">
                        <label for="company" class="form-label">Company</label>
                        <input type="text" class="form-control" id="company" required>
                    </div>
                    <div class="col-md-4">
                        <label for="year" class="form-label">Year</label>
                        <input type="number" class="form-control" id="year" min="1900" max="2024" required>
                    </div>
                    <div class="col-md-4">
                        <label for="kms_driven" class="form-label">Kilometers Driven</label>
                        <input type="number" class="form-control" id="kms_driven" min="0" required>
                    </div>
                    <div class="col-md-4">
                        <label for="fuel_type" class="form-label">Fuel Type</label>
                        <select class="form-select" id="fuel_type" required>
                            <option value="">Select fuel type</option>
                            <option value="Petrol">Petrol</option>
                            <option value="Diesel">Diesel</option>
                            <option value="CNG">CNG</option>
                        </select>
                    </div>
                </div>
                <div class="text-center mt-4">
                    <button type="submit" class="btn btn-primary btn-lg">Predict Price</button>
                </div>
            </form>

            <div class="loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Calculating price...</p>
            </div>

            <div class="result-container">
                <h3 class="text-center">Predicted Price</h3>
                <p class="text-center h2 text-primary" id="predictedPrice"></p>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.querySelector('.loading');
                const resultContainer = document.querySelector('.result-container');
                const predictedPrice = document.getElementById('predictedPrice');
                
                loading.style.display = 'block';
                resultContainer.style.display = 'none';
                
                const formData = {
                    name: document.getElementById('name').value,
                    company: document.getElementById('company').value,
                    year: document.getElementById('year').value,
                    kms_driven: document.getElementById('kms_driven').value,
                    fuel_type: document.getElementById('fuel_type').value
                };
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        predictedPrice.textContent = `₹${data.predicted_price.toLocaleString()}`;
                        resultContainer.style.display = 'block';
                    } else {
                        alert('Error: ' + data.error);
                    }
                } catch (error) {
                    alert('Error making prediction: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
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
        }), 500

# Vercel serverless function handler
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Route to the appropriate handler
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(home().encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Route to the appropriate handler
        if path == '/api/predict':
            # Get the request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Create a mock request object
            class MockRequest:
                def __init__(self, data):
                    self.json_data = data
                
                def get_json(self):
                    return self.json_data
            
            # Call the predict function
            response = predict(MockRequest(data))
            
            # Send the response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response.get_json()).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found') 