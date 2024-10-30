from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read CSV file
        df = pd.read_csv(file)
        
        # Return columns list initially
        return jsonify({
            'columns': df.columns.tolist(),
            'message': 'Please select a target column'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'No target column specified'}), 400

        # Read CSV file
        df = pd.read_csv(file)
        
        # Basic feature importance
        importances = []
        for col in df.columns:
            if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    corr = abs(df[col].astype(float).corr(df[target_column].astype(float)))
                    if not np.isnan(corr):
                        importances.append({
                            'feature': col,
                            'importance': float(corr * 100)
                        })
                except:
                    continue
        
        # Basic drift detection
        drifts = []
        mid = len(df) // 2
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    series = df[col].astype(float)
                    mean1 = float(series[:mid].mean())
                    mean2 = float(series[mid:].mean())
                    drift = abs(mean1 - mean2) / (abs(mean1) + 1e-10)
                    drifts.append({
                        'column': col,
                        'drift_detected': drift > 0.1,
                        'drift_score': float(drift * 100),
                        'p_value': float(1 - drift),
                        'stattest': 'mean_diff'
                    })
                except:
                    continue

        return jsonify({
            'message': 'Success',
            'feature_importances': sorted(importances, key=lambda x: x['importance'], reverse=True)[:5],
            'drift_scores': drifts
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()