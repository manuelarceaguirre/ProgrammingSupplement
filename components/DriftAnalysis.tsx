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
  statistic: number;
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
      <div className="overflow-x-auto">
        <table className="min-w-full text-gray-300">
          <thead>
            <tr>
              <th className="px-4 py-2">Column</th>
              <th className="px-4 py-2">Drift Score</th>
              <th className="px-4 py-2">Test Details</th>
              <th className="px-4 py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {driftScores.map((score, index) => (
              <tr key={index} className="border-t border-gray-700">
                <td className="px-4 py-2">{score.column}</td>
                <td className="px-4 py-2" style={{ color: score.color }}>
                  {score.drift_score.toFixed(2)}%
                </td>
                <td className="px-4 py-2">
                  {score.test_type}
                  <div className="text-sm text-gray-400">
                    Test Statistic: {score.statistic.toFixed(4)}
                    <br />
                    p-value: {score.p_value.toFixed(4)}
                  </div>
                </td>
                <td className="px-4 py-2" style={{ color: score.color }}>
                  {score.drift_detected ? 'Drift Detected' : 'No Drift'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-8 text-gray-300">
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