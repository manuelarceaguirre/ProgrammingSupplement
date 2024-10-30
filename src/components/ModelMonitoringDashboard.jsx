import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Alert, AlertTitle, AlertDescription } from './ui/alert';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const ModelMonitoringDashboard = () => {
  const [uploadStatus, setUploadStatus] = useState(null);
  const [featureImportances, setFeatureImportances] = useState([
    { feature: "Upload a CSV file to see feature importances", importance: 0 }
  ]);
  const [driftScores, setDriftScores] = useState([]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Using relative path for API endpoint
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      console.log('Response:', data); // Debug logging
      
      if (response.ok) {
        setUploadStatus({ type: 'success', message: data.message });
        setFeatureImportances(data.feature_importances);
        setDriftScores(data.drift_scores);
      } else {
        setUploadStatus({ 
          type: 'error', 
          message: data.error || 'Upload failed' 
        });
      }
    } catch (error) {
      console.error('Upload error:', error); // Debug logging
      setUploadStatus({ 
        type: 'error', 
        message: `Upload failed: ${error.message || 'Unknown error'}`
      });
    }
  };

  return (
    <div className="space-y-6 p-4">
      <h1 className="text-2xl font-bold mb-4">Credit Score Model Monitoring Dashboard</h1>
      
      {/* File Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>Upload CSV Data</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="block w-full text-sm text-slate-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-violet-50 file:text-violet-700
                hover:file:bg-violet-100"
            />
            {uploadStatus && (
              <Alert className={uploadStatus.type === 'success' ? 'border-green-400' : 'border-red-400'}>
                <AlertTitle>{uploadStatus.message}</AlertTitle>
              </Alert>
            )}
            <p className="text-sm text-gray-500">
              Maximum file size: 15MB. Only CSV files are accepted.
            </p>
          </div>
        </CardContent>
      </Card>
      
      {/* Feature Importance Section */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportances} layout="vertical">
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={150} />
                <Tooltip />
                <Bar dataKey="importance" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Drift Analysis Section */}
      {driftScores.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Data Drift Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {driftScores.map((score) => (
                <Alert key={score.column} className={score.drift_detected ? "border-red-400" : "border-green-400"}>
                  <AlertTitle className="flex justify-between">
                    <span>{score.column}</span>
                    <span className={`px-2 py-1 rounded-full text-sm ${
                      score.drift_detected ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"
                    }`}>
                      {score.drift_detected ? "Drift Detected" : "No Drift"}
                    </span>
                  </AlertTitle>
                  <AlertDescription>
                    <div className="grid grid-cols-3 gap-4 mt-2">
                      <div>
                        <span className="font-medium">Test:</span> {score.stattest}
                      </div>
                      <div>
                        <span className="font-medium">p-value:</span> {score.p_value.toFixed(3)}
                      </div>
                      <div>
                        <span className="font-medium">Drift Score:</span> {score.drift_score.toFixed(3)}
                      </div>
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Section */}
      {featureImportances.length > 0 && driftScores.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Model Monitoring Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-sm">
                <span className="font-medium">Top Feature:</span> {featureImportances[0].feature} (importance: {featureImportances[0].importance.toFixed(1)})
              </p>
              <p className="text-sm">
                <span className="font-medium">Drift Status:</span> {" "}
                {driftScores.some(s => s.drift_detected) 
                  ? "Significant drift detected in some features"
                  : "No significant drift detected"}
              </p>
              <p className="text-sm">
                <span className="font-medium">Recommendation:</span> {" "}
                {driftScores.some(s => s.drift_detected)
                  ? "Model retraining may be needed due to detected drift in key features"
                  : "Model performance appears stable"}
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ModelMonitoringDashboard;