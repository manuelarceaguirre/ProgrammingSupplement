import { NextApiRequest, NextApiResponse } from 'next'
import { MLDriftMonitor } from '@/api/drift_monitor'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { feature, startTime, endTime } = req.query
  const monitor = new MLDriftMonitor()
  
  // You'll need to implement loading of your reference and current data here
  // monitor.detect_drift(reference_data, current_data, feature_types)
  
  const history = monitor.get_drift_history(
    feature as string,
    startTime as string,
    endTime as string
  )
  
  res.status(200).json(history)
} 