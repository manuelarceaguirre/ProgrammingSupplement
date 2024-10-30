from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_feature_importance(df):
    # This is a simplified example - in real world, you might want to use
    # actual ML model feature importances
    importances = []
    for column in df.select_dtypes(include=[np.number]).columns:
        importance = abs(df[column].corr(df['Credit_Score'])) if 'Credit_Score' in df.columns else np.random.rand()
        importances.append({
            "feature": column,
            "importance": float(importance * 100)
        })
    return sorted(importances, key=lambda x: x['importance'], reverse=True)[:5]

def calculate_drift(df):
    # Simplified drift detection example
    drift_scores = []
    for column in df.select_dtypes(include=[np.number]).columns:
        # Split data in half to simulate reference and current datasets
        mid = len(df) // 2
        reference = df[column][:mid]
        current = df[column][mid:]
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference, current)
        
        drift_scores.append({
            "column": column,
            "drift_detected": p_value < 0.05,
            "p_value": float(p_value),
            "stattest": "ks",
            "drift_score": float(statistic)
        })
    return drift_scores[:3]  # Return top 3 drifting features

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded file
        try:
            df = pd.read_csv(filepath)
            feature_importances = calculate_feature_importance(df)
            drift_scores = calculate_drift(df)
            
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'filename': filename,
                'feature_importances': feature_importances,
                'drift_scores': drift_scores
            }), 200
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "API is running!"})

@app.route('/api')
def home():
    return "API is running"

if __name__ == '__main__':
    app.run(debug=True)