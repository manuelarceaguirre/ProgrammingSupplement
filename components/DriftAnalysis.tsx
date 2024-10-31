import { Tooltip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

interface DriftScore {
  column: string;
  drift_detected: boolean;
  drift_score: number;
  p_value: number;
  color: string;
  test_type: string;
  threshold: number;
}

interface TestDescriptions {
  [key: string]: string;
}

export default function DriftAnalysis({ 
  driftScores, 
  testDescriptions 
}: { 
  driftScores: DriftScore[];
  testDescriptions: TestDescriptions;
}) {
  return (
    <div className="mt-4">
      <h2 className="text-xl font-bold mb-4">Drift Analysis Results</h2>
      
      {/* Drift Scores */}
      {driftScores.map((score, index) => (
        <div key={index} style={{ color: score.color }} className="mb-4 p-4 border rounded">
          <strong>{score.column}</strong>
          <div>
            Test Type: {score.test_type}
            <Tooltip title={testDescriptions[score.test_type]}>
              <InfoIcon fontSize="small" className="ml-2 cursor-pointer" />
            </Tooltip>
          </div>
          <div>
            Drift Score: {score.drift_score.toFixed(2)}%
            {score.drift_detected && (
              <span style={{ color: 'red' }}>
                {' '}(Exceeds threshold of {(score.threshold * 100).toFixed(2)}%)
              </span>
            )}
          </div>
          <div>P-value: {score.p_value.toFixed(4)}</div>
        </div>
      ))}

      {/* Test Descriptions */}
      <div className="mt-8">
        <h3 className="text-lg font-bold mb-2">Test Descriptions:</h3>
        {Object.entries(testDescriptions).map(([test, description]) => (
          <div key={test} className="mb-2">
            <strong>{test}:</strong> {description}
          </div>
        ))}
      </div>
    </div>
  );
} 