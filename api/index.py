from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
import traceback

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    print(traceback.format_exc())
    response = jsonify({
        'error': str(error),
        'details': traceback.format_exc()
    })
    response.headers.add('Content-Type', 'application/json')
    return response, 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column', '')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # If no target column specified, just return the columns
        if not target_column:
            return jsonify({
                'columns': df.columns.tolist(),
                'message': 'Please select a target column'
            }), 200
        
        # Calculate feature importances and drift scores
        feature_importances = calculate_feature_importance(df, target_column)
        drift_scores = calculate_drift(df)
        
        response = jsonify({
            'message': 'File processed successfully',
            'feature_importances': feature_importances,
            'drift_scores': drift_scores
        })
        response.headers.add('Content-Type', 'application/json')
        return response, 200
        
    except Exception as e:
        return handle_error(e)

def calculate_feature_importance(df, target_column=None):
    try:
        importances = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_column:
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        # Process numeric features
        for column in numeric_cols:
            try:
                if target_column and target_column in df.columns:
                    if df[target_column].dtype in [np.number]:
                        correlation = abs(df[column].corr(df[target_column]))
                        importance = correlation * 100
                    else:
                        groups = [group[column].values for name, group in df.groupby(target_column)]
                        f_stat, _ = stats.f_oneway(*groups)
                        importance = min(f_stat * 10, 100) if not np.isnan(f_stat) else 0
                else:
                    mean = df[column].mean()
                    importance = (df[column].std() / mean * 100) if mean != 0 else 0
                
                importances.append({
                    "feature": str(column),
                    "importance": float(min(max(importance, 0), 100))
                })
            except Exception as e:
                print(f"Error processing numeric column {column}: {str(e)}")
                continue
        
        # Process categorical features
        for column in categorical_cols:
            try:
                if target_column and target_column in df.columns:
                    if df[target_column].dtype in [np.number]:
                        groups = [group[target_column].values for name, group in df.groupby(column)]
                        f_stat, _ = stats.f_oneway(*groups)
                        importance = min(f_stat * 10, 100) if not np.isnan(f_stat) else 0
                    else:
                        contingency = pd.crosstab(df[column], df[target_column])
                        chi2, _, _ = stats.chi2_contingency(contingency)
                        importance = min(chi2 * 10, 100)
                else:
                    value_counts = df[column].value_counts(normalize=True)
                    entropy = stats.entropy(value_counts)
                    importance = min(entropy * 50, 100)
                
                importances.append({
                    "feature": str(column),
                    "importance": float(importance)
                })
            except Exception as e:
                print(f"Error processing categorical column {column}: {str(e)}")
                continue
        
        return sorted(importances, key=lambda x: x['importance'], reverse=True)[:5]
    except Exception as e:
        print(f"Error in calculate_feature_importance: {str(e)}")
        raise

def calculate_drift(df):
    try:
        drift_scores = []
        midpoint = len(df) // 2
        
        for column in df.columns:
            try:
                if df[column].dtype in [np.number]:
                    # For numeric columns
                    part1 = df[column].iloc[:midpoint]
                    part2 = df[column].iloc[midpoint:]
                    statistic, p_value = stats.ks_2samp(part1, part2)
                    test_name = "Kolmogorov-Smirnov"
                else:
                    # For categorical columns
                    contingency = pd.crosstab(
                        df.index >= midpoint,
                        df[column]
                    )
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                    statistic = chi2
                    test_name = "Chi-square"
                
                drift_scores.append({
                    "column": str(column),
                    "drift_detected": bool(p_value < 0.05),
                    "p_value": float(p_value),
                    "drift_score": float(statistic),
                    "stattest": test_name
                })
            except Exception as e:
                print(f"Error processing drift for column {column}: {str(e)}")
                continue
                
        return drift_scores
    except Exception as e:
        print(f"Error in calculate_drift: {str(e)}")
        raise

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)