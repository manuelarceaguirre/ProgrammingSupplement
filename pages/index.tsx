import DriftAnalysis from '../components/DriftAnalysis';

// ... in your main component
const [driftScores, setDriftScores] = useState([]);
const [testDescriptions, setTestDescriptions] = useState({});

// ... in your handleSubmit or wherever you process the API response
const response = await fetch('/api/process', ...);
const data = await response.json();
setDriftScores(data.drift_scores);
setTestDescriptions(data.test_descriptions);

// ... in your render method
return (
  <div>
    {/* ... other components ... */}
    <DriftAnalysis 
      driftScores={driftScores} 
      testDescriptions={testDescriptions} 
    />
    {/* ... other components ... */}
  </div>
); 