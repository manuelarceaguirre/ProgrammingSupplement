from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from collections import defaultdict
from .drift_monitor import MLDriftMonitor

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
        
        # Process in chunks to manage memory
        chunk_size = 5000
        monitor = MLDriftMonitor()
        results = {}
        
        # Initialize feature types from first chunk
        train_chunk = pd.read_csv(train_file, nrows=1)
        feature_types = {}
        
        for col in train_chunk.columns:
            if col == target_column:
                feature_types[col] = 'target'
            elif pd.api.types.is_numeric_dtype(train_chunk[col]):
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'

        # Reset file pointers
        train_file.seek(0)
        test_file.seek(0)

        # Process data in chunks
        for train_chunk, test_chunk in zip(
            pd.read_csv(train_file, chunksize=chunk_size),
            pd.read_csv(test_file, chunksize=chunk_size)
        ):
            # Sample data if chunks are too large, but don't sample more than available
            sample_size = min(1000, len(train_chunk), len(test_chunk))
            if sample_size < len(train_chunk):
                train_chunk = train_chunk.sample(n=sample_size, random_state=42)
            if sample_size < len(test_chunk):
                test_chunk = test_chunk.sample(n=sample_size, random_state=42)

            chunk_results = monitor.detect_drift(train_chunk, test_chunk, feature_types)
            
            # Aggregate results
            for feature, result in chunk_results.items():
                if feature not in results:
                    results[feature] = []
                results[feature].append(result)

        # Average results across chunks
        final_results = []
        for feature, chunk_results in results.items():
            if not chunk_results:  # Skip if no results for this feature
                continue
                
            avg_severity = np.mean([r['severity'] for r in chunk_results])
            avg_statistic = np.mean([r['statistic'] for r in chunk_results])
            drift_detected = any(r['drift_detected'] for r in chunk_results)
            
            final_results.append({
                'column': feature,
                'drift_detected': drift_detected,
                'drift_score': float(avg_severity * 100),
                'p_value': float(1 - avg_severity),
                'statistic': float(avg_statistic),
                'stattest': feature_types[feature]
            })

        # Calculate feature importance for numerical columns
        importances = []
        for col in train_chunk.columns:
            if col != target_column and feature_types.get(col) == 'numerical':
                try:
                    corr = abs(train_chunk[col].corr(train_chunk[target_column]))
                    if not np.isnan(corr):
                        importances.append({
                            'feature': col,
                            'importance': float(corr * 100)
                        })
                except:
                    continue

        return jsonify({
            'message': 'Success',
            'feature_importances': sorted(importances, key=lambda x: x['importance'], reverse=True),
            'drift_scores': final_results
        })

    except Exception as e:
        print(f"Error in process_files: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Full error traceback
        return jsonify({'error': str(e)}), 500

# Error handler for 500 errors
@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    app.run()