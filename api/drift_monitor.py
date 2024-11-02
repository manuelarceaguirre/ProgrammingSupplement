import numpy as np
import pandas as pd
from typing import Dict, Union
from datetime import datetime

class MLDriftMonitor:
    """
    ML Drift Monitor implementing 3 essential tests without scipy dependency
    """
    
    def __init__(self):
        # Add history tracking
        self.drift_history = {
            'timestamps': [],
            'metrics': {}
        }
    
    def ks_test(self, reference: np.ndarray, current: np.ndarray, 
                threshold: float = 0.05) -> Dict[str, Union[bool, float]]:
        """
        Custom implementation of Kolmogorov-Smirnov test for numerical features.
        """
        # Sort the data
        ref_sorted = np.sort(reference)
        curr_sorted = np.sort(current)
        
        # Calculate empirical CDFs
        n_ref = len(reference)
        n_curr = len(current)
        ref_cdf = np.arange(1, n_ref + 1) / n_ref
        curr_cdf = np.arange(1, n_curr + 1) / n_curr
        
        # Interpolate to have same number of points
        all_values = np.sort(np.concatenate([ref_sorted, curr_sorted]))
        ref_cdf_interp = np.interp(all_values, ref_sorted, ref_cdf)
        curr_cdf_interp = np.interp(all_values, curr_sorted, curr_cdf)
        
        # Calculate KS statistic
        ks_statistic = np.max(np.abs(ref_cdf_interp - curr_cdf_interp))
        
        # Approximate critical value
        critical_value = np.sqrt(-0.5 * np.log(threshold/2)) * np.sqrt(
            (n_ref + n_curr) / (n_ref * n_curr)
        )
        
        return {
            'statistic': float(ks_statistic),
            'critical_value': float(critical_value),
            'drift_detected': bool(ks_statistic > critical_value),
            'severity': float(ks_statistic / critical_value)
        }
    
    def cramers_v_test(self, reference: np.ndarray, current: np.ndarray, 
                      threshold: float = 0.15) -> Dict[str, Union[bool, float]]:
        """
        Cramér's V test for categorical features.
        """
        # Get unique categories and create contingency table
        categories = np.unique(np.concatenate([reference, current]))
        n_categories = len(categories)
        
        # Create contingency table
        contingency = np.zeros((2, n_categories))
        for i, cat in enumerate(categories):
            contingency[0, i] = np.sum(reference == cat)
            contingency[1, i] = np.sum(current == cat)
        
        # Calculate chi-square statistic
        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        n = contingency.sum()
        expected = np.outer(row_sums, col_sums) / n
        
        # Handle zero expected frequencies
        mask = expected > 0
        chi_square = np.sum(
            np.where(mask,
                    (contingency - expected) ** 2 / expected,
                    0)
        )
        
        # Calculate Cramér's V
        min_dim = min(2, n_categories) - 1
        v = np.sqrt(chi_square / (n * min_dim)) if n * min_dim > 0 else 0
        
        return {
            'statistic': float(v),
            'threshold': float(threshold),
            'drift_detected': bool(v > threshold),
            'severity': float(v / threshold)
        }
    
    def psi_test(self, reference: np.ndarray, current: np.ndarray, 
                 bins: int = 10, threshold: float = 0.2) -> Dict[str, Union[bool, float]]:
        """
        Population Stability Index (PSI) for target/prediction drift.
        """
        def calculate_psi(ref_percent: np.ndarray, curr_percent: np.ndarray) -> float:
            epsilon = 1e-6
            ref_percent = np.maximum(epsilon, ref_percent)
            curr_percent = np.maximum(epsilon, curr_percent)
            return np.sum((curr_percent - ref_percent) * np.log(curr_percent / ref_percent))
        
        if len(np.unique(reference)) <= bins:
            # Categorical data
            categories = np.unique(np.concatenate([reference, current]))
            ref_hist = np.array([np.mean(reference == cat) for cat in categories])
            curr_hist = np.array([np.mean(current == cat) for cat in categories])
        else:
            # Numerical data
            edges = []
            for i in range(bins + 1):
                edge = np.percentile(reference, i * (100 / bins))
                edges.append(edge)
            edges[-1] += 1e-6
            
            ref_hist = np.zeros(bins)
            curr_hist = np.zeros(bins)
            
            for i in range(bins):
                ref_hist[i] = np.sum((reference >= edges[i]) & (reference < edges[i + 1]))
                curr_hist[i] = np.sum((current >= edges[i]) & (current < edges[i + 1]))
        
        # Normalize
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)
        
        psi_value = calculate_psi(ref_hist, curr_hist)
        
        return {
            'statistic': float(psi_value),
            'threshold': float(threshold),
            'drift_detected': bool(psi_value > threshold),
            'severity': float(psi_value / threshold)
        }
    
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                    feature_types: Dict[str, str]) -> Dict[str, dict]:
        """
        Detect drift across all features and store historical results.
        """
        results = {}
        timestamp = datetime.now().isoformat()
        
        # Calculate drift for each feature
        for feature, feat_type in feature_types.items():
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
                
            ref_values = reference_data[feature].values
            curr_values = current_data[feature].values
            
            if feat_type == 'numerical':
                results[feature] = self.ks_test(ref_values, curr_values)
            elif feat_type == 'categorical':
                results[feature] = self.cramers_v_test(ref_values, curr_values)
            elif feat_type == 'target':
                results[feature] = self.psi_test(ref_values, curr_values)
        
        # Store historical results
        self.drift_history['timestamps'].append(timestamp)
        for feature, result in results.items():
            if feature not in self.drift_history['metrics']:
                self.drift_history['metrics'][feature] = []
            self.drift_history['metrics'][feature].append({
                'timestamp': timestamp,
                'severity': result['severity'],
                'drift_detected': result['drift_detected']
            })
            
        return results

    def get_drift_history(self, feature: str = None, 
                         start_time: str = None, 
                         end_time: str = None) -> Dict:
        """
        Get historical drift data for UI visualization.
        """
        if not self.drift_history['timestamps']:
            return []

        # Filter by time range if specified
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        history = []
        for idx, timestamp in enumerate(self.drift_history['timestamps']):
            current_time = datetime.fromisoformat(timestamp)
            
            if start and current_time < start:
                continue
            if end and current_time > end:
                continue

            entry = {'date': timestamp}
            
            # Add metrics for specific feature or all features
            if feature:
                if feature in self.drift_history['metrics']:
                    entry['drift'] = self.drift_history['metrics'][feature][idx]['severity']
            else:
                # Average severity across all features
                severities = [
                    metrics[idx]['severity'] 
                    for metrics in self.drift_history['metrics'].values()
                ]
                entry['drift'] = sum(severities) / len(severities)
            
            history.append(entry)
            
        return history