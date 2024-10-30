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
        
        # Calculate feature importance using correlations
        feature_importances = []
        for col in df.columns:
            if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    corr = abs(df[col].corr(df[target_column]))
                    feature_importances.append({
                        'feature': col,
                        'importance': float(corr * 100)
                    })
                except:
                    continue
        
        # Sort and get top 5
        feature_importances = sorted(
            feature_importances, 
            key=lambda x: x['importance'], 
            reverse=True
        )[:5]
        
        # Calculate simple drift scores
        drift_scores = []
        midpoint = len(df) // 2
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    part1 = df[col][:midpoint].mean()
                    part2 = df[col][midpoint:].mean()
                    drift = abs(part1 - part2) / max(abs(part1), 1)
                    drift_scores.append({
                        'column': col,
                        'drift_detected': drift > 0.1,
                        'p_value': float(1 - drift),
                        'drift_score': float(drift * 100),
                        'stattest': 'mean_diff'
                    })
                except:
                    continue

        return jsonify({
            'message': 'Success',
            'feature_importances': feature_importances,
            'drift_scores': drift_scores
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)