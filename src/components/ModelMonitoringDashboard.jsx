import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import './ModelMonitoringDashboard.css';

function ModelMonitoringDashboard() {
  const [trainFile, setTrainFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('charges');
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [columns, setColumns] = useState([]);

  useEffect(() => {
    const loadInsuranceFiles = async () => {
      try {
        const trainResponse = await fetch('/insurance_train.csv');
        const testResponse = await fetch('/insurance_test.csv');
        
        if (!trainResponse.ok || !testResponse.ok) {
          throw new Error('Failed to load insurance files');
        }

        const trainBlob = await trainResponse.blob();
        const testBlob = await testResponse.blob();

        const defaultTrainFile = new File([trainBlob], 'insurance_train.csv', { type: 'text/csv' });
        const defaultTestFile = new File([testBlob], 'insurance_test.csv', { type: 'text/csv' });

        setTrainFile(defaultTrainFile);
        setTestFile(defaultTestFile);

        const text = await trainBlob.text();
        const headers = text.split('\n')[0].split(',');
        setColumns(headers);

        const formData = new FormData();
        formData.append('train_file', defaultTrainFile);
        formData.append('test_file', defaultTestFile);
        formData.append('target_column', 'charges');

        const response = await fetch('/api/process', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (response.ok) {
          setResults(data);
        } else {
          throw new Error(data.error || 'Error processing insurance files');
        }
      } catch (err) {
        setError('Error loading insurance files: ' + err.message);
      }
    };

    loadInsuranceFiles();
  }, []);

  const handleTrainFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setTrainFile(file);
      setResults(null);
      setError('');
      
      const text = await file.text();
      const headers = text.split('\n')[0].split(',');
      setColumns(headers);
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

  const getFeatureImportanceChart = () => {
    if (!results?.feature_importances) return null;

    const chartData = {
      options: {
        chart: {
          type: 'bar',
          background: 'var(--bg-secondary)',
          foreColor: 'var(--text-primary)',
        },
        theme: {
          mode: 'dark'
        },
        xaxis: {
          categories: results.feature_importances.map(item => item.feature),
        },
        yaxis: {
          title: {
            text: 'Importance (%)'
          }
        },
        plotOptions: {
          bar: {
            horizontal: true
          }
        },
        colors: ['#646cff']
      },
      series: [{
        name: 'Feature Importance',
        data: results.feature_importances.map(item => item.importance)
      }]
    };

    return (
      <ReactApexChart
        options={chartData.options}
        series={chartData.series}
        type="bar"
        height={350}
      />
    );
  };

  const getDriftChart = () => {
    if (!results?.drift_scores) return null;

    const chartData = {
      options: {
        chart: {
          type: 'bar',
          background: 'var(--bg-secondary)',
          foreColor: 'var(--text-primary)',
        },
        theme: {
          mode: 'dark'
        },
        xaxis: {
          categories: results.drift_scores.map(item => item.column),
        },
        yaxis: {
          title: {
            text: 'Drift Score (%)'
          }
        },
        plotOptions: {
          bar: {
            columnWidth: '50%',
            distributed: true, // Enable individual column colors
          }
        },
        colors: results.drift_scores.map(item => 
          item.drift_detected ? '#ff4d4d' : '#4caf50'  // Red if drift detected, green if not
        )
      },
      series: [{
        name: 'Drift Score',
        data: results.drift_scores.map(item => item.drift_score)
      }]
    };

    return (
      <ReactApexChart
        options={chartData.options}
        series={chartData.series}
        type="bar"
        height={350}
      />
    );
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

        <div className="target-selection">
          <h3>Target Column</h3>
          <select 
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="select-input"
          >
            <option value="">Select target column</option>
            {columns.map((col, index) => (
              <option key={index} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleProcess}
          disabled={!trainFile || !testFile || !targetColumn || loading}
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
            {getFeatureImportanceChart()}
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
            {getDriftChart()}
            <table className="table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Drift Score</th>
                  <th>Test Details</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {results.drift_scores.map((item, index) => (
                  <tr key={index}>
                    <td>{item.column}</td>
                    <td>{item.drift_score.toFixed(2)}%</td>
                    <td>
                      {item.test_type}
                      <div className="test-details">
                        <small>Test Statistic: {item.statistic.toFixed(4)}</small>
                        <br />
                        <small>p-value: {item.p_value.toFixed(4)}</small>
                      </div>
                    </td>
                    <td className={item.drift_detected ? 'drift-alert' : 'drift-ok'}>
                      {item.drift_detected ? 'Drift Detected' : 'No Drift'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {results.test_descriptions && (
              <div className="test-descriptions">
                <h3>Test Descriptions</h3>
                {Object.entries(results.test_descriptions).map(([test, description]) => (
                  <div key={test} className="test-description">
                    <strong>{test}:</strong> {description}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelMonitoringDashboard;