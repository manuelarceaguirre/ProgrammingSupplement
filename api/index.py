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
        print(f"Error: {str(e)}")
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
            if col != target_column:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        corr = df[col].corr(df[target_column])
                        importances.append({
                            'feature': col,
                            'importance': abs(float(corr * 100)) if not np.isnan(corr) else 0
                        })
                except:
                    continue
        
        # Basic drift detection
        drifts = []
        mid = len(df) // 2
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean1 = df[col][:mid].mean()
                    mean2 = df[col][mid:].mean()
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
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)