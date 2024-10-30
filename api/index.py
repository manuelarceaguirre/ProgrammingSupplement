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
        
        # Return columns list
        return jsonify({
            'columns': df.columns.tolist(),
            'message': 'File uploaded successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_files():
    try:
        if 'train_file' not in request.files or 'test_file' not in request.files:
            return jsonify({'error': 'Both train and test files are required'}), 400

        train_file = request.files['train_file']
        test_file = request.files['test_file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'No target column specified'}), 400

        # Read CSV files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Validate columns match
        if not all(col in test_df.columns for col in train_df.columns):
            return jsonify({'error': 'Train and test files have different columns'}), 400

        # Calculate feature importance using correlation with target
        importances = []
        for col in train_df.columns:
            if col != target_column and pd.api.types.is_numeric_dtype(train_df[col]):
                try:
                    corr = abs(train_df[col].astype(float).corr(train_df[target_column].astype(float)))
                    if not np.isnan(corr):
                        importances.append({
                            'feature': col,
                            'importance': float(corr * 100)
                        })
                except:
                    continue

        # Calculate drift between train and test
        drifts = []
        for col in train_df.columns:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                try:
                    train_mean = float(train_df[col].astype(float).mean())
                    test_mean = float(test_df[col].astype(float).mean())
                    train_std = float(train_df[col].astype(float).std())
                    
                    if train_std == 0:
                        drift_score = 0
                    else:
                        drift_score = abs(train_mean - test_mean) / train_std
                    
                    drifts.append({
                        'column': col,
                        'drift_detected': drift_score > 0.3,  # Threshold for drift detection
                        'drift_score': float(drift_score * 100),
                        'p_value': float(1 / (1 + drift_score)),  # Simplified p-value calculation
                        'stattest': 'mean_diff_normalized'
                    })
                except:
                    continue

        return jsonify({
            'message': 'Success',
            'feature_importances': sorted(importances, key=lambda x: x['importance'], reverse=True),
            'drift_scores': drifts
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()