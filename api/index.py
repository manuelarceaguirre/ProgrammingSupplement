from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 35 * 1024 * 1024  # Setting to 35MB to ensure 28.85MB files work

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_native_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    target_column = request.form.get('target_column', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        df = pd.read_csv(file)
        feature_importances = calculate_feature_importance(df, target_column if target_column else None)
        drift_scores = calculate_drift(df)
        
        return jsonify({
            'message': 'File processed successfully',
            'feature_importances': feature_importances,
            'drift_scores': drift_scores
        }), 200
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def calculate_feature_importance(df, target_column=None):
    importances = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter out the target column from numeric_cols if it exists
    if target_column:
        numeric_cols = [col for col in numeric_cols if col != target_column]
    
    for column in numeric_cols:
        if target_column and target_column in df.columns:
            # Calculate correlation-based importance with target
            correlation = abs(df[column].corr(df[target_column]))
            importance = correlation * 100
        else:
            # If no target, use variance as importance
            importance = (df[column].std() / df[column].mean() * 100) if df[column].mean() != 0 else 0
            
        importances.append({
            "feature": str(column),
            "importance": float(min(max(importance, 0), 100))
        })
    
    # Sort by importance and get top 5
    return sorted(importances, key=lambda x: x['importance'], reverse=True)[:5]

def calculate_drift(df):
    drift_scores = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Split data into two parts
    mid_point = len(df) // 2
    for column in numeric_cols:
        part1 = df[column][:mid_point]
        part2 = df[column][mid_point:]
        
        # Calculate basic statistics
        mean_diff = abs(part1.mean() - part2.mean())
        std_diff = abs(part1.std() - part2.std())
        
        # Normalize the differences
        mean_norm = mean_diff / part1.mean() if part1.mean() != 0 else mean_diff
        std_norm = std_diff / part1.std() if part1.std() != 0 else std_diff
        
        # Calculate drift score (0 to 1)
        drift_score = float((mean_norm + std_norm) / 2)
        
        # Calculate p-value using simple distribution comparison
        total_diff = abs(part1.mean() - part2.mean()) / (part1.std() + 1e-10)
        p_value = float(1 / (1 + np.exp(total_diff)))  # Convert to probability-like score
        
        drift_scores.append({
            "column": str(column),
            "drift_detected": bool(drift_score > 0.1),  # Convert numpy.bool_ to Python bool
            "p_value": p_value,
            "stattest": "mean_std_comparison",
            "drift_score": drift_score
        })
    
    # Sort by drift score and return top 3
    return sorted(drift_scores, key=lambda x: x['drift_score'], reverse=True)[:3]

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200, {'Content-Type': 'application/json'}

@app.route('/api/columns', methods=['POST'])
def get_columns():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read just the column names
        df = pd.read_csv(file, nrows=0)
        columns = df.columns.tolist()
        return jsonify({'columns': columns}), 200
    except Exception as e:
        return jsonify({'error': f'Error reading columns: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)