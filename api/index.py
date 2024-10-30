from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from collections import defaultdict

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
        
        # Read CSVs in chunks
        chunk_size = 10000  # Adjust based on your memory constraints
        train_df = pd.read_csv(train_file, chunksize=chunk_size)
        test_df = pd.read_csv(test_file, chunksize=chunk_size)

        # Process first chunk to validate columns and setup feature types
        train_chunk = next(train_df)
        test_chunk = next(test_df)
        
        if not all(col in test_chunk.columns for col in train_chunk.columns):
            return jsonify({'error': 'Train and test files have different columns'}), 400

        # Initialize drift monitor
        monitor = MLDriftMonitor()

        # Process only numeric and most important categorical columns
        important_columns = set()
        for col in train_chunk.columns:
            if col == target_column:
                important_columns.add(col)
            elif pd.api.types.is_numeric_dtype(train_chunk[col]):
                important_columns.add(col)
            elif train_chunk[col].nunique() < 20:  # Only process categorical cols with few unique values
                important_columns.add(col)

        # Process chunks
        results = defaultdict(list)
        for train_chunk, test_chunk in zip(train_df, test_df):
            train_chunk = train_chunk[list(important_columns)]
            test_chunk = test_chunk[list(important_columns)]
            
            # Process each chunk
            for col in important_columns:
                if len(results[col]) >= 3:  # Limit samples per column
                    continue
                    
                if pd.api.types.is_numeric_dtype(train_chunk[col]):
                    # Sample data to reduce computation
                    train_sample = train_chunk[col].sample(min(1000, len(train_chunk)))
                    test_sample = test_chunk[col].sample(min(1000, len(test_chunk)))
                    
                    result = monitor.ks_test(train_sample.values, test_sample.values)
                    results[col].append(result)

        # Average results across chunks
        final_results = []
        for col, chunk_results in results.items():
            if not chunk_results:
                continue
                
            avg_severity = np.mean([r['severity'] for r in chunk_results])
            drift_detected = any(r['drift_detected'] for r in chunk_results)
            
            final_results.append({
                'column': col,
                'drift_detected': drift_detected,
                'drift_score': float(avg_severity * 100),
                'p_value': float(1 - avg_severity),
                'stattest': 'numerical' if pd.api.types.is_numeric_dtype(train_chunk[col]) else 'categorical'
            })

        return jsonify({
            'message': 'Success',
            'drift_scores': final_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()