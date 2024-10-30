import React from 'react';
import './App.css';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import ModelMonitoringDashboard from './components/ModelMonitoringDashboard';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#646cff',
    },
    background: {
      default: '#1a1a1a',
      paper: '#2d2d2d',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <ModelMonitoringDashboard />
    </ThemeProvider>
  );
}

export default App;