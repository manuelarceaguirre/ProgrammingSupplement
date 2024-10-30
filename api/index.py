from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp  # Import only what we need

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = '/tmp'  # Use /tmp for serverless
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_feature_importance(df):
    importances = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        if 'Credit_Score' in df.columns:
            importance = abs(df[column].corr(df['Credit_Score']))
        else:
            importance = np.random.rand()
        importances.append({
            "feature": column,
            "importance": float(importance * 100)
        })
    return sorted(importances, key=lambda x: x['importance'], reverse=True)[:5]

def calculate_drift(df):
    drift_scores = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        mid = len(df) // 2
        reference = df[column][:mid].values
        current = df[column][mid:].values
        
        statistic, p_value = ks_2samp(reference, current)
        
        drift_scores.append({
            "column": column,
            "drift_detected": p_value < 0.05,
            "p_value": float(p_value),
            "stattest": "ks",
            "drift_score": float(statistic)
        })
    return drift_scores[:3]

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
        
        try:
            df = pd.read_csv(filepath)
            feature_importances = calculate_feature_importance(df)
            drift_scores = calculate_drift(df)
            
            # Clean up the file after processing
            os.remove(filepath)
            
            return jsonify({
                'message': 'File processed successfully',
                'feature_importances': feature_importances,
                'drift_scores': drift_scores
            }), 200
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)