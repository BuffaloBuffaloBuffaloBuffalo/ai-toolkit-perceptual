'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface IdentityPreview {
  path: string;
  step: number;
  t: number;
  /** ArcFace cosine similarity (generated x0 vs. reference); can be negative. */
  cos: number;
  srcName: string;
}

export default function useIdentityPreviews(jobID: string, reloadInterval: null | number = null) {
  const [previews, setPreviews] = useState<IdentityPreview[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refresh = () => {
    setStatus('loading');
    apiClient
      .get(`/api/jobs/${jobID}/identity-previews`)
      .then(res => res.data)
      .then(data => {
        if (data.previews) setPreviews(data.previews);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching identity previews:', error);
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
