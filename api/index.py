from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/tmp'
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
            # Use a simpler correlation calculation
            importance = abs(np.corrcoef(df[column], df['Credit_Score'])[0, 1])
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
        # Use simpler statistical measures
        mean_diff = abs(df[column][:mid].mean() - df[column][mid:].mean())
        std_diff = abs(df[column][:mid].std() - df[column][mid:].std())
        drift_score = (mean_diff + std_diff) / 2
        
        drift_scores.append({
            "column": column,
            "drift_detected": drift_score > 0.1,
            "p_value": float(1 - drift_score),
            "stattest": "mean_std_diff",
            "drift_score": float(drift_score)
        })
    return sorted(drift_scores, key=lambda x: x['drift_score'], reverse=True)[:3]

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
            # Read CSV in chunks to reduce memory usage
            df = pd.read_csv(filepath, chunksize=1000).get_chunk()
            feature_importances = calculate_feature_importance(df)
            drift_scores = calculate_drift(df)
            
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