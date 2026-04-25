'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';

export interface LossBreakdownSample {
  value: number;
  t?: number;
  sample?: string;
}

export interface LossBreakdown {
  samples: LossBreakdownSample[];
  n: number;
  mean: number | null;
  std: number | null;
}

export interface LossPoint {
  step: number;
  wall_time?: number;
  value: number | null;
  // Per-sample breakdown payload emitted by SDTrainer's MetricBuffer for
  // select metrics (e.g. id_sim, depth_loss, body_proportion_loss). Only
  // present on points where the trainer collected per-sample data; legacy
  // runs and metrics without per-sample collection have no `breakdown`.
  breakdown?: LossBreakdown;
}

type SeriesMap = Record<string, LossPoint[]>;

function isGraphableKey(key: string) {
  // treat anything containing "loss", "grad_norm", or "face_token_norm" as a graphable series
  return /loss|grad_norm|face_token_norm|txt_token_norm|vision_token_norm|body_token_norm|timestep|id_sim|id_clean|shape_sim|bp_sim|bsh_sim|body_shape_cos|body_shape_l1|body_shape_gated|normal_cos|normal_loss|pure_noise|va_level|va_mid|va_edge/i.test(key);
}

export default function useJobLossLog(jobID: string, reloadInterval: null | number = null) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);

  // track last step per key so polling is incremental per series
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const lossKeys = useMemo(() => {
    const base = (keys ?? []).filter(isGraphableKey);
    // if keys table is empty early on, fall back to just "loss"
    if (base.length === 0) return ['loss'];
    return base.sort();
  }, [keys]);

  const refreshLoss = useCallback(async () => {
    if (!jobID) return;

    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
    setStatus(loadStatus);

    try {
      // Step 1: get key list (we can do this by calling endpoint once; it returns keys)
      // Keep it cheap: limit=1.
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: 'loss', limit: 1 } })
        .then(res => res.data as { keys?: string[] });

      const newKeys = first.keys ?? [];
      setKeys(newKeys);

      const wantedLossKeys = (newKeys.filter(isGraphableKey).length ? newKeys.filter(isGraphableKey) : ['loss']).sort();

      // Step 2: fetch each loss key incrementally (since_step per key if polling)
      const requests = wantedLossKeys.map(k => {
        const params: Record<string, any> = { key: k };

        if (reloadInterval && lastStepByKeyRef.current[k] != null) {
          // lastStep tracks our shifted (-1) step; the API filters by raw
          // sqlite step, so add 1 back to query incrementally.
          params.since_step = (lastStepByKeyRef.current[k] as number) + 1;
        }

        params.limit = 1000000;

        return apiClient
          .get(`/api/jobs/${jobID}/loss`, { params })
          .then(res => res.data as { key: string; points?: LossPoint[] });
      });

      const results = await Promise.all(requests);

      setSeries(prev => {
        const next: SeriesMap = { ...prev };

        for (const r of results) {
          const k = r.key;
          // Sqlite stores step as the trainer's loop index, which lands one
          // ahead of the tqdm progress display (tqdm "248/1000" shares its row
          // with sqlite step 249). Shift so the chart's "step N" lines up with
          // the tqdm log row the user reads.
          const newPoints = (r.points ?? [])
            .filter(p => p.value !== null)
            .map(p => ({ ...p, step: p.step - 1 }));

          if (!didInitialLoadRef.current) {
            // initial: replace
            next[k] = newPoints;
          } else if (newPoints.length) {
            const existing = next[k] ?? [];
            const prevLast = existing.length ? existing[existing.length - 1].step : null;
            const filtered = prevLast == null ? newPoints : newPoints.filter(p => p.step > prevLast);
            next[k] = filtered.length ? [...existing, ...filtered] : existing;
          } else {
            // no new points: keep existing
            next[k] = next[k] ?? [];
          }

          // update last step per key
          const finalArr = next[k] ?? [];
          lastStepByKeyRef.current[k] = finalArr.length
            ? finalArr[finalArr.length - 1].step
            : (lastStepByKeyRef.current[k] ?? null);
        }

        // remove stale loss keys that no longer exist (rare, but keeps UI clean)
        for (const existingKey of Object.keys(next)) {
          if (isGraphableKey(existingKey) && !wantedLossKeys.includes(existingKey)) {
            delete next[existingKey];
            delete lastStepByKeyRef.current[existingKey];
          }
        }

        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching loss logs:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval]);

  useEffect(() => {
    // reset when job changes
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus('idle');

    refreshLoss();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshLoss();
      }, reloadInterval);

      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refreshLoss]);

  return { series, keys, lossKeys, status, refreshLoss, setSeries };
}
