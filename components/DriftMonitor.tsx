import React, { useState, useEffect } from 'react'

export default function Component() {
  const [driftData, setDriftData] = useState([])
  const [timeframe, setTimeframe] = useState('day')

  useEffect(() => {
    const fetchDriftHistory = async () => {
      const response = await fetch('/api/drift-history')
      const data = await response.json()
      setDriftData(data)
    }
    
    fetchDriftHistory()
  }, [timeframe])

  // ... rest of your component code ...
} 