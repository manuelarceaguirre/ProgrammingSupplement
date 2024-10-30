import React, { useState } from 'react';
import './ModelMonitoringDashboard.css';

function ModelMonitoringDashboard() {
  const [trainFile, setTrainFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleTrainFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setTrainFile(file);
      // Reset results when new file is uploaded
      setResults(null);
      setError('');
      
      // Get columns from train file
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        if (response.ok) {
          setColumns(data.columns);
        } else {
          setError(data.error || 'Error uploading train file');
        }
      } catch (err) {
        setError('Error uploading train file');
      }
    }
  };

  const handleTestFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setTestFile(file);
      setResults(null);
      setError('');
    }
  };

  const handleProcess = async () => {
    if (!trainFile || !testFile) {
      setError('Please upload both train and test files');
      return;
    }

    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('train_file', trainFile);
    formData.append('test_file', testFile);
    formData.append('target_column', targetColumn);

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setResults(data);
      } else {
        setError(data.error || 'Error processing files');
      }
    } catch (err) {
      setError('Error processing files');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <div className="card">
        <h2>Data Upload</h2>
        
        <div className="upload-section">
          <div>
            <h3>Training Data</h3>
            <input
              type="file"
              accept=".csv"
              onChange={handleTrainFileUpload}
              className="file-input"
            />
            <p>{trainFile ? trainFile.name : 'No file selected'}</p>
          </div>

          <div>
            <h3>Test Data</h3>
            <input
              type="file"
              accept=".csv"
              onChange={handleTestFileUpload}
              className="file-input"
            />
            <p>{testFile ? testFile.name : 'No file selected'}</p>
          </div>
        </div>

        {columns.length > 0 && (
          <div className="target-selection">
            <h3>Select Target Column</h3>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="select"
            >
              <option value="">Select column...</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>
        )}

        <button
          onClick={handleProcess}
          disabled={loading || !trainFile || !testFile || !targetColumn}
          className="process-button"
        >
          {loading ? 'Processing...' : 'Process Files'}
        </button>

        {error && <p className="error">{error}</p>}
      </div>

      {results && (
        <div className="results-section">
          <div className="card">
            <h2>Feature Importance</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Importance</th>
                </tr>
              </thead>
              <tbody>
                {results.feature_importances.map((item, index) => (
                  <tr key={index}>
                    <td>{item.feature}</td>
                    <td>{item.importance.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h2>Data Drift Analysis</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Drift Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {results.drift_scores.map((item, index) => (
                  <tr key={index}>
                    <td>{item.column}</td>
                    <td>{item.drift_score.toFixed(2)}%</td>
                    <td className={item.drift_detected ? 'drift-alert' : 'drift-ok'}>
                      {item.drift_detected ? 'Drift Detected' : 'No Drift'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelMonitoringDashboard;