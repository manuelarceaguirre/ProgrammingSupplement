from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfRows, TestColumnsType, TestShareOfMissingValues
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import time
from joblib import parallel_backend
import warnings
from functools import partial
from concurrent.futures import TimeoutError
import signal
from datetime import datetime
import os
from evidently.pipeline.column_mapping import ColumnMapping
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

class EvidentlyAssistant:
    def __init__(self, data: pd.DataFrame, target: str, reference_size: float = 0.7, time_unit_column: Optional[str] = None):
        """Initialize the assistant with data and configuration"""
        self.data = data
        self.target = target
        self.reference_size = reference_size
        self.time_unit_column = time_unit_column  # Store the time unit column

        # Determine target type
        self.target_type = self._determine_target_type()

        # Split numerical and categorical columns
        self.numerical_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove time unit column from features if specified
        if self.time_unit_column:
            if self.time_unit_column in self.numerical_cols:
                self.numerical_cols.remove(self.time_unit_column)
            if self.time_unit_column in self.categorical_cols:
                self.categorical_cols.remove(self.time_unit_column)

        # Split reference and current datasets
        self._split_data()

        # Initialize label encoders dictionary
        self.label_encoders = {}

    def _determine_target_type(self) -> str:
        """Determine if target is categorical or numerical"""
        if self.data[self.target].dtype in ['object', 'category']:
            return 'classification'
        return 'regression'

    def _split_data(self):
        """Split data into reference and current datasets with proper column mapping"""
        try:
            # Calculate split point
            split_idx = int(len(self.data) * self.reference_size)

            # Split the data
            self.reference_data = self.data.iloc[:split_idx].copy()
            self.current_data = self.data.iloc[split_idx:].copy()

            # Create column mapping using Evidently's ColumnMapping class
            self.column_mapping = ColumnMapping(
                target=self.target,
                numerical_features=self.numerical_cols,
                categorical_features=self.categorical_cols,
                embeddings=None,  # Add if you have embeddings
                task='classification' if self.target_type == 'classification' else 'regression'
            )

            print(f"\nData split:")
            print(f"- Reference dataset size: {len(self.reference_data):,}")
            print(f"- Current dataset size: {len(self.current_data):,}")

        except Exception as e:
            print(f"Error splitting data: {str(e)}")
            raise

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for feature importance calculation"""
        df_encoded = df.copy()

        for col in self.categorical_cols + [self.target]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df_encoded

    def analyze_dataset(self) -> Dict:
        """Analyze dataset structure and characteristics"""
        try:
            num_categorical = len(self.categorical_cols)
            num_numerical = len(self.numerical_cols)

            print(f"Found {num_categorical} categorical and {num_numerical} numerical columns")
            print(f"Data split: {{'reference_size': {len(self.reference_data)}, 'current_size': {len(self.current_data)}}}")

            return {
                'categorical_columns': self.categorical_cols,
                'numerical_columns': self.numerical_cols,
                'total_columns': len(self.data.columns),
                'total_rows': len(self.data),
                'reference_size': len(self.reference_data),
                'current_size': len(self.current_data)
            }
        except Exception as e:
            print(f"Error analyzing dataset: {str(e)}")
            raise

    def set_target(self, target_column: str) -> Dict:
        """Set target variable and analyze it"""
        if target_column not in self.data.columns:
            raise ValueError(f"Column {target_column} not found in dataset")

        self.target = target_column

        # Remove target from feature lists
        if target_column in self.categorical_cols:
            self.categorical_cols.remove(target_column)
            target_type = 'categorical'
        if target_column in self.numerical_cols:
            self.numerical_cols.remove(target_column)
            target_type = 'numerical'

        # Create column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=self.numerical_cols,
            categorical_features=self.categorical_cols
        )

        # Analyze target
        target_report = Report(metrics=[
            TargetDriftPreset(),
        ])

        target_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        # Get target metrics safely
        try:
            target_metrics = target_report.as_dict()
        except:
            target_metrics = {"error": "Could not extract target metrics"}

        # Get target distribution
        target_distribution = {
            'reference': self.reference_data[target_column].value_counts().to_dict(),
            'current': self.current_data[target_column].value_counts().to_dict()
        }

        return {
            'target_column': target_column,
            'target_type': target_type,
            'target_metrics': target_metrics,
            'target_distribution': target_distribution,
            'feature_split': {
                'categorical_features': len(self.categorical_cols),
                'numerical_features': len(self.numerical_cols)
            }
        }

    def analyze_features(self) -> Dict:
        """Analyze features and their characteristics"""
        try:
            feature_analysis = {}

            for column in self.data.columns:
                if column != self.target and column != self.time_unit_column:
                    feature_info = {
                        'type': 'numerical' if column in self.numerical_cols else 'categorical',
                        'unique_values': len(self.data[column].unique()),
                        'missing_values': self.data[column].isnull().sum(),
                        'cardinality': len(self.data[column].unique()) / len(self.data),
                    }

                    # Add numerical statistics if applicable
                    if column in self.numerical_cols:
                        feature_info.update({
                            'mean': float(self.data[column].mean()),
                            'std': float(self.data[column].std()),
                            'min': float(self.data[column].min()),
                            'max': float(self.data[column].max())
                        })

                    feature_analysis[column] = feature_info

            return feature_analysis

        except Exception as e:
            print(f"Error analyzing features: {str(e)}")
            raise

    def get_feature_importance(self, methods: List[str] = None) -> Dict:
        """Calculate feature importance using different methods with progress tracking"""
        if methods is None:
            methods = ['random_forest']

        print("\nPreparing data...")
        # Encode categorical variables
        data_encoded = self._encode_categorical_variables(self.data)

        # Filter out ID columns and other non-meaningful features
        excluded_patterns = ['id', 'ID', 'identifier', 'index']
        features_to_use = [col for col in data_encoded.columns
                          if not any(pattern.lower() in col.lower()
                                     for pattern in excluded_patterns)]

        # Remove target and time unit column from features
        if self.target in features_to_use:
            features_to_use.remove(self.target)
        if self.time_unit_column and self.time_unit_column in features_to_use:
            features_to_use.remove(self.time_unit_column)

        # Prepare data
        X = data_encoded[features_to_use]
        y = data_encoded[self.target]
        feature_names = X.columns.tolist()

        importance_results = {}

        # Create progress bar for methods
        with tqdm(total=len(methods), desc="Methods", position=0) as pbar:
            for method in methods:
                try:
                    start_time = time.time()

                    if method == 'random_forest':
                        pbar.set_description(f"Random Forest")
                        if self.target_type == 'classification':
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)

                        with tqdm(total=2, desc="Steps", position=1, leave=False) as inner_pbar:
                            inner_pbar.set_description("Training model")
                            model.fit(X, y)
                            inner_pbar.update(1)

                            inner_pbar.set_description("Calculating importance")
                            importance_values = model.feature_importances_
                            inner_pbar.update(1)

                        importance_results['random_forest'] = {
                            'importance_values': importance_values.tolist(),
                            'feature_names': feature_names,
                            'sorted_features': [x for _, x in sorted(
                                zip(importance_values, feature_names),
                                reverse=True
                            )],
                            'computation_time': f"{time.time() - start_time:.2f} seconds"
                        }

                    # Add other methods here if selected
                    elif method == 'permutation':
                        pbar.set_description(f"Permutation")
                        # Implement permutation importance calculation
                        # ...

                    elif method == 'shap':
                        pbar.set_description(f"SHAP")
                        # Implement SHAP values calculation
                        # ...

                    elif method == 'mutual_info':
                        pbar.set_description(f"Mutual Information")
                        # Implement mutual information calculation
                        # ...

                except Exception as e:
                    importance_results[method] = {
                        'error': str(e),
                        'computation_time': f"{time.time() - start_time:.2f} seconds"
                    }

                pbar.update(1)

        return importance_results

    def test_features(self) -> Dict:
        """Run statistical tests on features"""
        test_suite = TestSuite(tests=[
            # Basic data tests
            TestNumberOfRows(),
            TestColumnsType(),
            TestShareOfMissingValues(),
        ])

        test_suite.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        # Create drift report separately
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])

        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        try:
            return {
                'test_suite_results': test_suite.as_dict(),
                'drift_analysis': drift_report.as_dict()
            }
        except Exception as e:
            return {"error": f"Could not extract test results: {str(e)}"}

    def save_powerbi_format(self, results_dict: Dict, output_dir: str):
      """Save results in Power BI friendly CSV format"""
      rows = []
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      test_name = self.target.lower()

      # Extract feature-specific results
      feature_results = results_dict.get('feature_analysis', {})
      importance_results = results_dict.get('feature_importance', {})
      drift_results = results_dict.get('drift_results', {})

      for feature in self.data.columns:
          try:
              # Initialize base row with feature metadata
              base_row = {
                  'test_name': test_name,
                  'analysis_timestamp': timestamp,
                  'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                  'analysis_time': datetime.now().strftime("%H:%M:%S"),
                  'feature_name': feature,
                  'feature_type': 'numerical' if feature in self.numerical_cols else 'categorical',
                  'unique_values': len(self.reference_data[feature].unique()),
                  'missing_values': self.reference_data[feature].isnull().sum(),
                  'unique_ratio': len(self.reference_data[feature].unique()) / len(self.reference_data),
                  'is_target': feature == self.target,
                  'target_variable': self.target,
                  'target_type': self.target_type,
                  'total_features': len(self.data.columns),
                  'numerical_features': len(self.numerical_cols),
                  'categorical_features': len(self.categorical_cols),
                  'reference_size': len(self.reference_data),
                  'current_size': len(self.current_data),
                  'time_unit_column': self.time_unit_column,  # Include time unit column name
              }

              # Add time unit data summary for the feature
              if self.time_unit_column and self.time_unit_column != feature:
                  time_unit_values = self.data[self.time_unit_column]
                  base_row['time_unit_min'] = time_unit_values.min()
                  base_row['time_unit_max'] = time_unit_values.max()
                  base_row['time_unit_unique_values'] = time_unit_values.nunique()
              else:
                  base_row['time_unit_min'] = None
                  base_row['time_unit_max'] = None
                  base_row['time_unit_unique_values'] = None

              # Add drift results
              if feature in drift_results:
                  feature_drift = drift_results[feature]
                  for test_name, test_results in feature_drift.items():
                      if isinstance(test_results, dict):
                          base_row[f'{test_name}_drift_score'] = test_results.get('drift_score')
                          base_row[f'{test_name}_p_value'] = test_results.get('p_value')
                          base_row[f'{test_name}_is_drifted'] = test_results.get('is_drifted', False)

                  # Set overall drift status
                  is_drifted = any(test.get('is_drifted', False)
                                  for test in feature_drift.values()
                                  if isinstance(test, dict))
                  base_row['drift_status'] = 'Drifted' if is_drifted else 'No Drift'

              # Add importance scores
              if 'random_forest' in importance_results:
                  if feature in importance_results['random_forest']['feature_names']:
                      idx = importance_results['random_forest']['feature_names'].index(feature)
                      importance_value = importance_results['random_forest']['importance_values'][idx]
                      base_row['random_forest_importance'] = importance_value
                      base_row['random_forest_rank'] = idx + 1  # Rank starts from 1

              rows.append(base_row)

          except Exception as e:
              print(f"Error processing {feature}: {str(e)}")
              continue

      # Save to CSV
      output_path = f"{output_dir}/feature_analysis_{test_name}_{timestamp}.csv"
      pd.DataFrame(rows).to_csv(output_path, index=False)
      print(f"\nPower BI friendly file saved as: {output_path}")


    def calculate_drift_scores(self) -> Dict:
        """Calculate drift scores with individual column test selection"""
        drift_results = {}
        total_columns = len(self.data.columns) - 1  # Excluding target and time unit column
        if self.time_unit_column:
            total_columns -= 1
        current_column = 0

        print("\nCalculating drift scores column by column...")

        for column in self.data.columns:
            if column == self.target or column == self.time_unit_column:
                continue

            current_column += 1
            is_numerical = column in self.numerical_cols
            drift_results[column] = {}

            print(f"\nColumn {current_column}/{total_columns}: {column}")

            if is_numerical:
                print("Available tests for this numerical column:")
                print("1. Kolmogorov-Smirnov test (ks)")
                print("2. Wasserstein Distance (wasserstein)")
                print("3. Population Stability Index (psi)")
                print("4. Skip this column")
            else:
                print("Available tests for this categorical column:")
                print("1. Chi-square test (chisquare)")
                print("2. Population Stability Index (psi)")
                print("3. Jensen-Shannon Distance (jensenshannon)")
                print("4. Skip this column")

            choice = input("Select test (1-4): ").strip()

            try:
                if choice != '4':  # Not skipping
                    if is_numerical:
                        if choice == '1':
                            self._run_numerical_test(column, 'ks', drift_results)
                        elif choice == '2':
                            self._run_numerical_test(column, 'wasserstein', drift_results)
                        elif choice == '3':
                            self._run_numerical_test(column, 'psi', drift_results)
                    else:
                        if choice == '1':
                            self._run_categorical_test(column, 'chisquare', drift_results)
                        elif choice == '2':
                            self._run_categorical_test(column, 'psi', drift_results)
                        elif choice == '3':
                            self._run_categorical_test(column, 'jensenshannon', drift_results)
                else:
                    print(f"Skipping column {column}")
            except Exception as e:
                print(f"Error calculating drift for {column}: {str(e)}")

        return drift_results

    def _run_numerical_test(self, column: str, test_name: str, results: Dict):
        """Helper method to run numerical drift tests with improved status tracking"""
        try:
            drift_metric = ColumnDriftMetric(
                column_name=column,
                stattest=test_name
            )
            drift_report = Report(metrics=[drift_metric])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=self.current_data,
                column_mapping=self.column_mapping
            )
            result = drift_report.as_dict()['metrics'][0]['result']

            # Store test results
            results[column][test_name] = {
                'drift_score': result.get('drift_score', None),
                'p_value': result.get('p_value', None),
                'is_drifted': result.get('drift_detected', False)
            }

            # Update drift status
            if results[column][test_name]['is_drifted']:
                results[column]['drift_status'] = 'Drifted'
                results[column]['drift_detection_method'] = test_name
            else:
                results[column]['drift_status'] = 'No Drift'

        except Exception as e:
            print(f"Error in {test_name} test for {column}: {str(e)}")
            results[column][test_name] = {
                'drift_score': None,
                'p_value': None,
                'is_drifted': False,
                'error': str(e)
            }

    def _run_categorical_test(self, column: str, test_name: str, results: Dict):
        """Helper method to run categorical drift tests with improved error handling"""
        try:
            drift_metric = ColumnDriftMetric(
                column_name=column,
                stattest=test_name
            )
            drift_report = Report(metrics=[drift_metric])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                drift_report.run(
                    reference_data=self.reference_data,
                    current_data=self.current_data,
                    column_mapping=self.column_mapping
                )

            result = drift_report.as_dict()['metrics'][0]['result']

            # Store test results
            results[column][test_name] = {
                'drift_score': result.get('drift_score', None),
                'p_value': result.get('p_value', None) if test_name != 'psi' else None,
                'is_drifted': result.get('drift_detected', False)
            }

            # Update drift status
            if results[column][test_name]['is_drifted']:
                results[column]['drift_status'] = 'Drifted'
                results[column]['drift_detection_method'] = test_name
            else:
                results[column]['drift_status'] = 'No Drift'

        except Exception as e:
            print(f"Error in {test_name} test for {column}: {str(e)}")
            results[column][test_name] = {
                'drift_score': None,
                'p_value': None,
                'is_drifted': False,
                'error': str(e)
            }

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # 1. Dataset structure analysis
            print("\n1. Analyzing dataset structure...")
            dataset_analysis = self.analyze_dataset()

            # 2. Set target variable
            print("\n2. Setting target variable...")
            print(f"Target variable: {self.target} ({self.target_type})")

            # 3. Feature analysis
            print("\n3. Analyzing features...")
            feature_analysis = self.analyze_features()
            print("Feature analysis complete")

            # 4. Feature importance
            print("\n4. Calculating feature importance...")
            # Prompt user to select feature importance methods
            importance_results = self.calculate_feature_importance()

            # 5. Drift analysis
            print("\n5. Calculating drift scores...")
            drift_results = self.calculate_drift_scores()

            # 6. Statistical tests
            print("\n6. Running statistical tests...")
            statistical_results = self.run_statistical_tests()
            print("Statistical tests complete")

            # Combine all results
            results = {
                'dataset_analysis': dataset_analysis,
                'target_analysis': {'target': self.target, 'type': self.target_type},
                'feature_analysis': feature_analysis,
                'feature_importance': importance_results,
                'drift_results': drift_results,
                'statistical_tests': statistical_results
            }

            return results

        except Exception as e:
            print(f"Error in analysis pipeline: {str(e)}")
            raise

    def calculate_feature_importance(self):
        """Calculate feature importance using selected methods"""
        try:
            print("\nFeature Importance Methods:")
            print("1. Random Forest (faster, built-in importance)")
            print("2. Permutation Importance (slower, model agnostic)")
            print("3. SHAP Values (slower, more detailed)")
            print("4. Mutual Information (faster, statistical measure)")
            print("5. Done selecting")

            selected_methods = []
            while True:
                choice = input("\nSelect a method (1-5): ")
                if choice == '5':
                    break
                elif choice in ['1', '2', '3', '4']:
                    method_name = {
                        '1': 'random_forest',
                        '2': 'permutation',
                        '3': 'shap',
                        '4': 'mutual_info'
                    }[choice]
                    if method_name not in selected_methods:
                        selected_methods.append(method_name)
                        print(f"Selected methods: {selected_methods}")
                    else:
                        print("Method already selected.")
                else:
                    print("Invalid choice. Please select 1-5.")

            if not selected_methods:
                print("No methods selected. Using default method: Random Forest")
                selected_methods = ['random_forest']

            print(f"\nCalculating importance using: {selected_methods}")
            importance_results = self.get_feature_importance(methods=selected_methods)
            return importance_results

        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            raise

    def run_statistical_tests(self):
        """Run statistical tests on the data"""
        try:
            statistical_results = {}

            # 1. Distribution tests
            for column in self.numerical_cols:
                if column != self.target:
                    # Kolmogorov-Smirnov test for distribution comparison
                    ref_data = self.reference_data[column].dropna()
                    curr_data = self.current_data[column].dropna()

                    if len(ref_data) > 0 and len(curr_data) > 0:
                        ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)

                        statistical_results[column] = {
                            'ks_test': {
                                'statistic': float(ks_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        }

            # 2. Correlation analysis
            ref_corr = self.reference_data[self.numerical_cols].corr()
            curr_corr = self.current_data[self.numerical_cols].corr()

            correlation_changes = {}
            for col1 in self.numerical_cols:
                for col2 in self.numerical_cols:
                    if col1 < col2:  # Only look at unique pairs
                        ref_val = ref_corr.loc[col1, col2]
                        curr_val = curr_corr.loc[col1, col2]
                        change = abs(ref_val - curr_val)

                        if change > 0.1:  # Report significant correlation changes
                            correlation_changes[f"{col1}_vs_{col2}"] = {
                                'reference_correlation': float(ref_val),
                                'current_correlation': float(curr_val),
                                'absolute_change': float(change)
                            }

            # 3. Basic statistics comparison
            for column in self.numerical_cols:
                if column not in statistical_results:
                    statistical_results[column] = {}

                ref_stats = self.reference_data[column].describe()
                curr_stats = self.current_data[column].describe()

                statistical_results[column]['basic_stats'] = {
                    'reference': {
                        'mean': float(ref_stats['mean']),
                        'std': float(ref_stats['std']),
                        'min': float(ref_stats['min']),
                        'max': float(ref_stats['max']),
                        'median': float(ref_stats['50%'])
                    },
                    'current': {
                        'mean': float(curr_stats['mean']),
                        'std': float(curr_stats['std']),
                        'min': float(curr_stats['min']),
                        'max': float(curr_stats['max']),
                        'median': float(curr_stats['50%'])
                    }
                }

            # 4. Categorical variables distribution comparison
            for column in self.categorical_cols:
                if column != self.target:
                    ref_counts = self.reference_data[column].value_counts()
                    curr_counts = self.current_data[column].value_counts()

                    # Align categories and normalize
                    all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                    ref_counts = ref_counts.reindex(all_categories).fillna(0)
                    curr_counts = curr_counts.reindex(all_categories).fillna(0)

                    # Convert to proportions
                    ref_props = ref_counts / ref_counts.sum()
                    curr_props = curr_counts / curr_counts.sum()

                    # Calculate chi-square statistic manually
                    expected = ref_props * len(self.current_data)
                    observed = curr_counts

                    valid_mask = expected > 0  # Only use categories present in reference
                    if valid_mask.sum() > 0:
                        chi2_stat = np.sum(((observed[valid_mask] - expected[valid_mask]) ** 2) / expected[valid_mask])
                        df = valid_mask.sum() - 1
                        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

                        statistical_results[column] = {
                            'distribution_test': {
                                'statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(df),
                                'significant': p_value < 0.05
                            }
                        }

            return {
                'numerical_tests': statistical_results,
                'correlation_changes': correlation_changes
            }

        except Exception as e:
            print(f"Error running statistical tests: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Get data path from user
        data_path = input("Enter path to your data file: ")
        data = pd.read_csv(data_path)

        # Show columns and allow dropping
        while True:
            print("\nAvailable columns:")
            for i, col in enumerate(data.columns, 1):
                print(f"{i}. {col}")

            print("\nOptions:")
            print("1. Drop columns")
            print("2. Continue with analysis")
            choice = input("\nSelect option (1-2): ")

            if choice == "1":
                drop_cols = input("\nEnter column numbers to drop (comma-separated, e.g., 1,2,3): ")
                try:
                    # Convert input to column indices
                    drop_indices = [int(x.strip()) - 1 for x in drop_cols.split(",")]
                    # Get column names to drop
                    cols_to_drop = [data.columns[i] for i in drop_indices]
                    # Drop columns
                    data = data.drop(columns=cols_to_drop)
                    print(f"\nDropped columns: {', '.join(cols_to_drop)}")
                except Exception as e:
                    print(f"Error dropping columns: {str(e)}")
            else:
                break

        # Ask for the time unit column after dropping columns
        print("\nPlease specify the time unit column (if any) from the remaining columns.")
        print("If there is no time unit column, just press Enter.")
        print("\nAvailable columns:")
        for i, col in enumerate(data.columns, 1):
            print(f"{i}. {col}")
        time_unit_input = input("\nEnter the number corresponding to the time unit column or press Enter to skip: ")
        time_unit_column = None
        if time_unit_input.strip() != "":
            time_unit_idx = int(time_unit_input) - 1
            time_unit_column = data.columns[time_unit_idx]
            print(f"\nTime unit column set to: {time_unit_column}")

        # Get target variable from user
        print("\nSelect target variable:")
        for i, col in enumerate(data.columns, 1):
            print(f"{i}. {col}")
        target_idx = int(input("\nEnter number: ")) - 1
        target = data.columns[target_idx]

        # Initialize assistant with time unit column
        assistant = EvidentlyAssistant(
            data=data,
            target=target,
            reference_size=0.7,
            time_unit_column=time_unit_column
        )

        # Run analysis
        results = assistant.run_analysis()

        # Save results
        output_dir = "evidently_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results using NumpyEncoder
        with open(f"{output_dir}/analysis_results.json", 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        print(f"\nResults saved to {output_dir}/analysis_results.json")

        # Save Power BI format
        assistant.save_powerbi_format(results, output_dir)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
