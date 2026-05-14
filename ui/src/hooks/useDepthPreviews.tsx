'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface DepthPreview {
  path: string;
  kind: 'image' | 'video';
  step: number;
  t: number;
  dc?: number;
  srcName?: string;
}

export default function useDepthPreviews(jobID: string, reloadInterval: null | number = null) {
  const [previews, setPreviews] = useState<DepthPreview[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refresh = () => {
    setStatus('loading');
    apiClient
      .get(`/api/jobs/${jobID}/depth-previews`)
      .then(res => res.data)
      .then(data => {
        if (data.previews) setPreviews(data.previews);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching depth previews:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refresh();
    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID]);

  return { previews, status, refresh };
}
