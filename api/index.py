from flask import Flask,jsonify,request;from flask_cors import CORS;import pandas as pd;import numpy as np
from collections import defaultdict;from .drift_monitor import MLDriftMonitor
#I was limited by Vercel's free version so I included few libraries

app = Flask(__name__)
CORS(app)#We create the Flask app and allow CORS requests for front and api communication
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
#We create the enpoint and check if the file was uploaded and valid
        df = pd.read_csv(file)
#Read CSV file        
        return jsonify({
            'columns': df.columns.tolist(),
            'message': 'File uploaded successfully'
        })
#Gets the columns and delivers a succesful message
    except Exception as e:
        return jsonify({'error': str(e)}), 500 #returns exceptions that occur in the upload
@app.route('/api/process', methods=['POST']) #sets endpoint to process both train and test files
def process_files():
    try:
        if 'train_file' not in request.files or 'test_file' not in request.files:
            return jsonify({'error': 'Both train and test files are required'}), 400
        train_file = request.files['train_file']
        test_file = request.files['test_file']
        target_column = request.form.get('target_column')#defines test, train and the target column
        chunk_size = 5000
        monitor = MLDriftMonitor()
        results = {}
        importances = []
#I had to process in chunks to manage memory, creates a monitor object, dict for results and the list for ft importance
        train_chunk = pd.read_csv(train_file, nrows=1)#Reads the first row
        feature_types = {}
        for col in train_chunk.columns:#Iterates on columns to determine if they are target, numerical, or categorical
            if col == target_column:
                feature_types[col] = 'target'
            elif pd.api.types.is_numeric_dtype(train_chunk[col]):
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'
        train_file.seek(0)
        test_file.seek(0)
#It resets file pointers to start over the reading
        first_train_chunk = pd.read_csv(train_file, nrows=chunk_size) #Reads the first chunk for ft importance
        for col in first_train_chunk.columns:
            if col != target_column and feature_types.get(col) == 'numerical':
                try:
                    corr = abs(first_train_chunk[col].corr(first_train_chunk[target_column]))
                    if not np.isnan(corr):
                        importances.append({
                            'feature': col,
                            'importance': float(corr * 100)
                        }) #Calculates ft importance by doing absolute correlation to numerical
                except:
                    continue
        train_file.seek(0) #We reset again
        for train_chunk, test_chunk in zip(
            pd.read_csv(train_file, chunksize=chunk_size),
            pd.read_csv(test_file, chunksize=chunk_size)
        ):
            sample_size = min(1000, len(train_chunk), len(test_chunk))
            if sample_size < len(train_chunk):
                train_chunk = train_chunk.sample(n=sample_size, random_state=42)
            if sample_size < len(test_chunk):
                test_chunk = test_chunk.sample(n=sample_size, random_state=42)
            chunk_results = monitor.detect_drift(train_chunk, test_chunk, feature_types)
#It samples data and uses detect_drift
            for feature, result in chunk_results.items():
                if feature not in results:
                    results[feature] = []
                results[feature].append(result)
#Aggregates by each column
        final_results = []
        for feature, chunk_results in results.items():
            if not chunk_results:
                continue
            avg_severity = np.mean([r['severity'] for r in chunk_results])
            avg_statistic = np.mean([r['statistic'] for r in chunk_results])
            drift_detected = any(r['drift_detected'] for r in chunk_results)
#It calculates average severity, average statistic, and checks if drift was detected
            test_type = {
                'numerical': 'Kolmogorov-Smirnov Test',
                'categorical': 'Cramér\'s V Test',
                'target': 'Population Stability Index'
            }.get(feature_types[feature], 'Unknown Test') #Decide which test is best
            final_results.append({
                'column': feature,
                'drift_detected': drift_detected,
                'drift_score': float(avg_severity * 100),
                'p_value': float(1 - avg_severity),
                'statistic': float(avg_statistic),
                'stattest': feature_types[feature],
                'test_type': test_type,
                'color': 'red' if drift_detected else 'green',
                'threshold': {
                    'numerical': 0.60,  # 60% threshold
                    'categorical': 0.70,  # 70% threshold
                    'target': 0.80,  # 80% threshold
                }.get(feature_types[feature], 0.60)
            })
#It appends all results with drift score, p value and threshold
        final_results.sort(key=lambda x: x['drift_score'], reverse=True)
        return jsonify({
            'message': 'Success',
            'feature_importances': sorted(importances, key=lambda x: x['importance'], reverse=True),
            'drift_scores': final_results,
            'test_descriptions': {
                'Kolmogorov-Smirnov Test': 'Measures the maximum distance between two cumulative distribution functions. Used for numerical features.',
                'Cramér\'s V Test': 'Measures the strength of association between categorical variables. Based on chi-square statistic.',
                'Population Stability Index': 'Measures how much a distribution has shifted between two samples. Used for target variables.'
            },
            'threshold_descriptions': {
                'numerical': 'Drift threshold: 60% (significant change)',
                'categorical': 'Drift threshold: 70% (significant change)',
                'target': 'Drift threshold: 80% (significant change)'
            }
        })
#Sorts all the results by drift score, feature importance and the test descriptions
    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500 #This part handles just the 500 error 
if __name__ == '__main__':
    app.run()#Starts the app