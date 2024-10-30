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
    
    # Calculate correlation with Credit_Score if it exists
    target = 'Credit_Score' if 'Credit_Score' in df.columns else None
    
    for column in numeric_cols:
        if column == target:
            continue
            
        if target:
            # Calculate correlation-based importance
            correlation = abs(df[column].corr(df[target]))
            importance = correlation * 100
        else:
            # If no target, use variance as importance
            importance = (df[column].std() / df[column].mean() * 100) if df[column].mean() != 0 else 0
            
        importances.append({
            "feature": column,
            "importance": float(min(max(importance, 0), 100))  # Ensure between 0-100
        })
    
    # Sort by importance and take top 5
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
        drift_score = (mean_norm + std_norm) / 2
        
        # Calculate p-value using simple distribution comparison
        total_diff = abs(part1.mean() - part2.mean()) / (part1.std() + 1e-10)
        p_value = 1 / (1 + np.exp(total_diff))  # Convert to probability-like score
        
        drift_scores.append({
            "column": column,
            "drift_detected": drift_score > 0.1,  # Threshold for drift detection
            "p_value": float(p_value),
            "stattest": "mean_std_comparison",
            "drift_score": float(drift_score)
        })
    
    # Sort by drift score and return top 3
    return sorted(drift_scores, key=lambda x: x['drift_score'], reverse=True)[:3]

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400, {'Content-Type': 'application/json'}
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400, {'Content-Type': 'application/json'}
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400, {'Content-Type': 'application/json'}
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV in chunks to reduce memory usage
            df = pd.read_csv(filepath)
            feature_importances = calculate_feature_importance(df)
            drift_scores = calculate_drift(df)
            
            os.remove(filepath)  # Clean up
            
            return jsonify({
                'message': 'File processed successfully',
                'feature_importances': feature_importances,
                'drift_scores': drift_scores
            }), 200, {'Content-Type': 'application/json'}
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500, {'Content-Type': 'application/json'}
            
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500, {'Content-Type': 'application/json'}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200, {'Content-Type': 'application/json'}

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large (max 15MB)'}), 413, {'Content-Type': 'application/json'}

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500, {'Content-Type': 'application/json'}

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)