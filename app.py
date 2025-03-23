# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('gold_price_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Load the dataset for display
gold_data = pd.read_csv('https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv')
gold_data['Date'] = pd.to_datetime(gold_data['Date'])
gold_data = gold_data.sort_values('Date')

@app.route('/')
def home():
    # Get some basic statistics
    latest_price = gold_data.iloc[-1]['Price']
    highest_price = gold_data['Price'].max()
    lowest_price = gold_data['Price'].min()
    avg_price = gold_data['Price'].mean()
    
    # Calculate year-over-year change
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Get data for the current year and previous year
    current_year_data = gold_data[
        (gold_data['Date'].dt.year == current_year - 1) & 
        (gold_data['Date'].dt.month == current_month)
    ]
    
    prev_year_data = gold_data[
        (gold_data['Date'].dt.year == current_year - 2) & 
        (gold_data['Date'].dt.month == current_month)
    ]
    
    if not current_year_data.empty and not prev_year_data.empty:
        yoy_change = ((current_year_data['Price'].values[0] - prev_year_data['Price'].values[0]) / 
                      prev_year_data['Price'].values[0] * 100)
    else:
        yoy_change = 0
    
    # Get recent data for display
    recent_data = gold_data.tail(24).to_dict('records')
    
    return render_template('index.html', 
                           latest_price=latest_price,
                           highest_price=highest_price,
                           lowest_price=lowest_price,
                           avg_price=avg_price,
                           yoy_change=yoy_change,
                           recent_data=recent_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        year = int(request.form['year'])
        month = int(request.form['month'])
        
        # Prepare features
        features = np.array([[year, month]])
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/data')
def data():
    # Convert dataframe to list of dictionaries for display
    data_list = gold_data.to_dict('records')
    
    # Format date for display
    for item in data_list:
        item['Date'] = item['Date'].strftime('%Y-%m')
    
    return render_template('data.html', data=data_list)

if __name__ == '__main__':
    # Make sure the static folder exists
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(debug=True)