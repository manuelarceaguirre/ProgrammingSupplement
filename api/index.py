from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from scipy import stats

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
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Remove target column from feature lists
    if target_column:
        numeric_cols = [col for col in numeric_cols if col != target_column]
        categorical_cols = [col for col in categorical_cols if col != target_column]
    
    for column in numeric_cols:
        if target_column and target_column in df.columns:
            if df[target_column].dtype in [np.number]:
                # For numeric target, use correlation
                correlation = abs(df[column].corr(df[target_column]))
                importance = correlation * 100
            else:
                # For categorical target, use ANOVA F-value
                groups = [group[column].values for name, group in df.groupby(target_column)]
                f_stat, _ = stats.f_oneway(*groups)
                importance = min(f_stat * 10, 100) if not np.isnan(f_stat) else 0
        else:
            # If no target, use coefficient of variation
            mean = df[column].mean()
            importance = (df[column].std() / mean * 100) if mean != 0 else 0
            
        importances.append({
            "feature": str(column),
            "importance": float(min(max(importance, 0), 100))
        })
    
    for column in categorical_cols:
        if target_column and target_column in df.columns:
            if df[target_column].dtype in [np.number]:
                # For numeric target, use ANOVA
                groups = [group[target_column].values for name, group in df.groupby(column)]
                f_stat, _ = stats.f_oneway(*groups)
                importance = min(f_stat * 10, 100) if not np.isnan(f_stat) else 0
            else:
                # For categorical target, use Chi-square
                contingency = pd.crosstab(df[column], df[target_column])
                chi2, _, _ = stats.chi2_contingency(contingency)
                importance = min(chi2 * 10, 100)
        else:
            # If no target, use entropy
            value_counts = df[column].value_counts(normalize=True)
            entropy = stats.entropy(value_counts)
            importance = min(entropy * 50, 100)
            
        importances.append({
            "feature": str(column),
            "importance": float(importance)
        })
    
    # Sort by importance and get top 5
    return sorted(importances, key=lambda x: x['importance'], reverse=True)[:5]

def calculate_drift(df):
    drift_scores = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Split data into two parts
    midpoint = len(df) // 2
    
    for column in numeric_cols:
        # For numeric columns, use Kolmogorov-Smirnov test
        part1 = df[column].iloc[:midpoint]
        part2 = df[column].iloc[midpoint:]
        
        statistic, p_value = stats.ks_2samp(part1, part2)
        
        drift_scores.append({
            "column": str(column),
            "drift_detected": p_value < 0.05,
            "p_value": float(p_value),
            "drift_score": float(statistic),
            "stattest": "Kolmogorov-Smirnov"
        })
    
    for column in categorical_cols:
        # For categorical columns, use Chi-square test
        contingency = pd.crosstab(
            df.index >= midpoint,  # True for second half
            df[column]
        )
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        drift_scores.append({
            "column": str(column),
            "drift_detected": p_value < 0.05,
            "p_value": float(p_value),
            "drift_score": float(chi2),
            "stattest": "Chi-square"
        })
    
    return drift_scores

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