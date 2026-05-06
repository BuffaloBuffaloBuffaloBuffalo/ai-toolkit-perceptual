'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';

// =====================================================================
// Step 4: legacy-to-canonical rename map.
//
// Mirrors `extensions_built_in/sd_trainer/metric_naming.py` so historical
// runs (where only the legacy key was logged) render under the new
// `subsystem/kind/variant` namespace in the new metrics tab.
//
// The new dashboard groups by the segment before the first `/`. Anything
// not in this map keeps its legacy name and falls into a "Custom" group.
// Keep in lock-step with the Python `CANONICAL_RENAMES` map. Every entry
// here MUST exist there (and vice versa) so dual-write and back-rendering
// stay symmetric.
// =====================================================================
export const LEGACY_TO_CANONICAL: Record<string, string> = {
  // core
  loss: 'core/loss',
  grad_norm: 'core/grad_norm',
  timestep: 'core/timestep',

  // diffusion
  diffusion_loss: 'diffusion/loss_raw',
  diffusion_loss_applied: 'diffusion/loss_applied',

  // identity
  identity_loss: 'identity/loss_raw',
  identity_loss_applied: 'identity/loss_applied',
  id_sim: 'identity/sim',
  id_clean_target: 'identity/clean_target',
  id_clean_delta: 'identity/clean_delta',

  // landmark
  landmark_loss: 'landmark/loss_raw',
  landmark_loss_applied: 'landmark/loss_applied',

  // body proportion
  body_proportion_loss: 'body_proportion/loss_raw',
  body_proportion_loss_applied: 'body_proportion/loss_applied',

  // body shape
  body_shape_loss: 'body_shape/loss_raw',
  body_shape_loss_applied: 'body_shape/loss_applied',
  body_shape_cos: 'body_shape/cos',
  body_shape_l1: 'body_shape/l1',
  body_shape_gated_pct: 'body_shape/gated_pct',

  // normals
  normal_loss: 'normal/loss_raw',
  normal_loss_applied: 'normal/loss_applied',
  normal_cos: 'normal/cos',

  // vae anchor
  vae_anchor_loss: 'vae_anchor/loss_raw',
  vae_anchor_loss_applied: 'vae_anchor/loss_applied',
  va_level_1: 'vae_anchor/level/level_1',
  va_level_2: 'vae_anchor/level/level_2',
  va_level_3: 'vae_anchor/level/level_3',
  va_mid: 'vae_anchor/level/mid',
  va_edge: 'vae_anchor/level/edge',

  // depth
  depth_consistency_loss: 'depth/loss_raw',
  depth_consistency_loss_applied: 'depth/loss_applied',
  depth_consistency_ssi: 'depth/ssi',
  depth_consistency_grad: 'depth/grad',

  // gradient cosine diagnostic
  grad_norm_diffusion: 'grad/norm/diffusion',
  grad_norm_depth: 'grad/norm/depth',
  grad_cos_diff_depth: 'grad/cos/diff_depth',

  // tokens
  face_token_norm: 'tokens/face/norm',
  vision_token_norm: 'tokens/vision/norm',
  body_token_norm: 'tokens/body/norm',
  txt_token_norm: 'tokens/text/norm',

  // aux
  pure_noise_cos: 'aux/pure_noise_cos',

  // legacy `loss/` prefixed keys (BaseSDTrainProcess used to wrap
  // anything starting with `loss` under a `loss/` namespace).
  'loss/loss': 'core/loss',
  'loss/grad_norm': 'core/grad_norm',
  'loss/diffusion_loss': 'diffusion/loss_raw',
  'loss/diffusion_loss_applied': 'diffusion/loss_applied',
  'loss/identity_loss': 'identity/loss_raw',
  'loss/identity_loss_applied': 'identity/loss_applied',
  'loss/landmark_loss': 'landmark/loss_raw',
  'loss/landmark_loss_applied': 'landmark/loss_applied',
  'loss/body_proportion_loss': 'body_proportion/loss_raw',
  'loss/body_proportion_loss_applied': 'body_proportion/loss_applied',
  'loss/body_shape_loss': 'body_shape/loss_raw',
  'loss/body_shape_loss_applied': 'body_shape/loss_applied',
  'loss/normal_loss': 'normal/loss_raw',
  'loss/normal_loss_applied': 'normal/loss_applied',
  'loss/vae_anchor_loss': 'vae_anchor/loss_raw',
  'loss/vae_anchor_loss_applied': 'vae_anchor/loss_applied',
  'loss/depth_consistency_loss': 'depth/loss_raw',
  'loss/depth_consistency_loss_applied': 'depth/loss_applied',

  // epoch averages
  'loss/epoch_avg': 'core/loss/epoch_avg',
  'loss/identity_loss_epoch_avg': 'identity/loss_raw/epoch_avg',
  'loss/diffusion_loss_epoch_avg': 'diffusion/loss_raw/epoch_avg',
  id_sim_epoch_avg: 'identity/sim/epoch_avg',
  'loss/body_proportion_loss_epoch_avg': 'body_proportion/loss_raw/epoch_avg',
  'loss/depth_consistency_loss_epoch_avg': 'depth/loss_raw/epoch_avg',
};

// Pattern-shaped renames (per-t-band bins). Match `<prefix>_t<NN>` where
// `<prefix>` is one of the legacy bin prefixes; canonical form is
// `<subsystem>/<kind>/t<NN>`.
const _BIN_PATTERN = /^([a-z_]+?)_t(\d{2,3})$/;
const _BIN_PREFIX_TO_CANONICAL: Record<string, string> = {
  id_sim: 'identity/sim',
  shape_sim: 'landmark/sim',
  bp_sim: 'body_proportion/sim',
  bsh_sim: 'body_shape/sim',
  depth_loss: 'depth/loss',
  diffusion_loss: 'diffusion/loss',
};

/** Map a legacy or canonical key to its canonical form. Returns the input
 * unchanged if no mapping exists (e.g. user-defined custom metrics). */
export function canonicalizeKey(key: string): string {
  if (LEGACY_TO_CANONICAL[key]) return LEGACY_TO_CANONICAL[key];
  const m = _BIN_PATTERN.exec(key);
  if (m) {
    const prefix = m[1];
    const n = m[2];
    const canonical = _BIN_PREFIX_TO_CANONICAL[prefix];
    if (canonical) return `${canonical}/t${n}`;
  }
  return key;
}

/** Subsystem grouping segment (everything before the first `/`). */
export function subsystemOf(key: string): string {
  const idx = key.indexOf('/');
  return idx === -1 ? 'custom' : key.slice(0, idx);
}

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
          params.since_step = lastStepByKeyRef.current[k];
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
          const newPoints = (r.points ?? []).filter(p => p.value !== null);

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

// =====================================================================
// Companion hook for the new metrics dashboard (step 5).
//
// Differences from `useJobLossLog`:
//   - No regex allowlist — fetches every key in the run except
//     `_meta/*` markers and `learning_rate` (which is rendered separately).
//   - Maps each legacy key to its canonical form via `canonicalizeKey`,
//     and merges duplicate-mapped series so legacy `loss` and
//     dual-written `core/loss` show as a single canonical `core/loss`.
//   - Preserves the `breakdown` field on each point so the new tooltip
//     can render the per-sample drilldown.
// =====================================================================
export function useJobMetricsLog(jobID: string, reloadInterval: null | number = null) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [rawKeys, setRawKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  // canonical keys derived from rawKeys
  const canonicalKeys = useMemo(() => {
    const set = new Set<string>();
    for (const k of rawKeys) {
      if (k.startsWith('_meta/')) continue;
      if (k === 'learning_rate') continue;
      set.add(canonicalizeKey(k));
    }
    return Array.from(set).sort();
  }, [rawKeys]);

  const refresh = useCallback(async () => {
    if (!jobID) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(didInitialLoadRef.current ? 'refreshing' : 'loading');

    try {
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: 'loss', limit: 1 } })
        .then(res => res.data as { keys?: string[] });

      const allKeys = (first.keys ?? []).filter(k => !k.startsWith('_meta/') && k !== 'learning_rate');
      setRawKeys(first.keys ?? []);

      // Fetch every legacy key, since each may carry distinct points (the
      // dual-write may have only kicked in mid-run). We then merge by
      // canonical name on the consumer side.
      const requests = allKeys.map(k => {
        const params: Record<string, any> = { key: k };
        if (reloadInterval && lastStepByKeyRef.current[k] != null) {
          params.since_step = lastStepByKeyRef.current[k];
        }
        params.limit = 1000000;
        return apiClient
          .get(`/api/jobs/${jobID}/loss`, { params })
          .then(res => res.data as { key: string; points?: LossPoint[] });
      });

      const results = await Promise.all(requests);

      setSeries(prev => {
        const next: SeriesMap = { ...prev };

        // Group fetched results by canonical key name. When the legacy
        // and canonical name resolve to the same canonical key, their
        // points are unioned by step so we don't draw two overlapping
        // lines for the same metric.
        const byCanonical: Record<string, LossPoint[]> = {};
        for (const r of results) {
          const canonical = canonicalizeKey(r.key);
          const newPoints = (r.points ?? []).filter(p => p.value !== null);
          (byCanonical[canonical] ||= []).push(...newPoints);

          // track last seen step on the legacy key (raw sqlite step) so
          // incremental polling stays incremental.
          if (newPoints.length) {
            const lastStep = newPoints[newPoints.length - 1].step;
            const prevLast = lastStepByKeyRef.current[r.key];
            if (prevLast == null || lastStep > prevLast) {
              lastStepByKeyRef.current[r.key] = lastStep;
            }
          }
        }

        for (const [canonical, pts] of Object.entries(byCanonical)) {
          // Merge with existing points by step. Later points (higher step)
          // win on collision.
          const existing = next[canonical] ?? [];
          const stepMap = new Map<number, LossPoint>();
          for (const p of existing) stepMap.set(p.step, p);
          for (const p of pts) stepMap.set(p.step, p);
          const merged = Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
          next[canonical] = merged;
        }

        // remove canonical keys that no longer have any underlying legacy key.
        const liveCanonical = new Set<string>();
        for (const k of allKeys) liveCanonical.add(canonicalizeKey(k));
        for (const k of Object.keys(next)) {
          if (!liveCanonical.has(k)) delete next[k];
        }

        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching metrics logs:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval]);

  useEffect(() => {
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setRawKeys([]);
    setStatus('idle');

    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refresh]);

  return { series, canonicalKeys, rawKeys, status, refresh, setSeries };
}

// =====================================================================
// Multi-job metrics fetch — fans out the canonical-key fetch logic from
// `useJobMetricsLog` across N jobs in parallel. Used by the cross-job
// comparison view.
//
// Returns:
//   seriesByJob: { [jobId]: SeriesMap }
//   canonicalKeysByJob: { [jobId]: string[] }
//   unionCanonicalKeys: string[]   — sorted union across all jobs
//   intersectCanonicalKeys: string[] — sorted intersection across jobs
//   status / refresh
// =====================================================================
export interface MultiJobMetricsState {
  seriesByJob: Record<string, SeriesMap>;
  canonicalKeysByJob: Record<string, string[]>;
  unionCanonicalKeys: string[];
  intersectCanonicalKeys: string[];
  status: 'idle' | 'loading' | 'success' | 'error' | 'refreshing';
  refresh: () => void;
}

export function useMultiJobMetricsLog(
  jobIDs: string[],
  reloadInterval: null | number = null,
): MultiJobMetricsState {
  const [seriesByJob, setSeriesByJob] = useState<Record<string, SeriesMap>>({});
  const [rawKeysByJob, setRawKeysByJob] = useState<Record<string, string[]>>({});
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  // per-job-per-key last step
  const lastStepByJobKeyRef = useRef<Record<string, Record<string, number | null>>>({});
  // monotonic counter used to discard results from stale in-flight refreshes
  // when the caller swaps the job set out from under us. Bumped synchronously
  // whenever jobIDsKey changes (via the latestJobIDsRef effect).
  const generationRef = useRef(0);
  // Always-fresh view of the caller's job set so refresh() never relies on
  // its closure value (which would go stale across job swaps).
  const latestJobIDsRef = useRef<string[]>(jobIDs);

  // Stabilise the dependency for useEffect/useCallback regardless of array
  // identity churn from the caller.
  const jobIDsKey = useMemo(() => jobIDs.slice().sort().join('|'), [jobIDs]);

  // Keep latestJobIDsRef + generation in sync with jobIDs synchronously on
  // each render so refresh() always sees the current set.
  if (latestJobIDsRef.current !== jobIDs) {
    const prevKey = latestJobIDsRef.current.slice().sort().join('|');
    if (prevKey !== jobIDsKey) generationRef.current += 1;
    latestJobIDsRef.current = jobIDs;
  }

  const canonicalKeysByJob = useMemo(() => {
    const out: Record<string, string[]> = {};
    for (const [jid, raw] of Object.entries(rawKeysByJob)) {
      const set = new Set<string>();
      for (const k of raw) {
        if (k.startsWith('_meta/')) continue;
        if (k === 'learning_rate') continue;
        set.add(canonicalizeKey(k));
      }
      out[jid] = Array.from(set).sort();
    }
    return out;
  }, [rawKeysByJob]);

  const unionCanonicalKeys = useMemo(() => {
    const set = new Set<string>();
    for (const arr of Object.values(canonicalKeysByJob)) {
      for (const k of arr) set.add(k);
    }
    return Array.from(set).sort();
  }, [canonicalKeysByJob]);

  const intersectCanonicalKeys = useMemo(() => {
    const arrays = Object.values(canonicalKeysByJob);
    if (arrays.length === 0) return [];
    let acc = new Set<string>(arrays[0]);
    for (let i = 1; i < arrays.length; i++) {
      const next = new Set<string>();
      for (const k of arrays[i]) if (acc.has(k)) next.add(k);
      acc = next;
    }
    return Array.from(acc).sort();
  }, [canonicalKeysByJob]);

  const refresh = useCallback(async () => {
    const currentJobIDs = latestJobIDsRef.current;
    if (currentJobIDs.length === 0) {
      setSeriesByJob({});
      setRawKeysByJob({});
      setStatus('success');
      didInitialLoadRef.current = true;
      return;
    }
    // Skip if a refresh is already mid-flight for the same generation.
    // Guarantees we don't pile dozens of parallel waves on top of each
    // other when the polling interval is short relative to fetch latency
    // (N keys × M jobs requests can saturate the browser connection pool).
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    const myGen = generationRef.current;
    setStatus(didInitialLoadRef.current ? 'refreshing' : 'loading');

    try {
      // Phase 1: discover keys for each job (cheap call, limit=1).
      const discovery = await Promise.all(
        currentJobIDs.map(jid =>
          apiClient
            .get(`/api/jobs/${jid}/loss`, { params: { key: 'loss', limit: 1 } })
            .then(res => ({ jid, keys: (res.data?.keys ?? []) as string[] }))
            .catch(() => ({ jid, keys: [] as string[] })),
        ),
      );

      // If our generation was bumped while Phase 1 was awaiting (the caller
      // changed the job set), drop the result. The fresh refresh will run.
      if (myGen !== generationRef.current) return;

      const rawKeysNext: Record<string, string[]> = {};
      for (const d of discovery) rawKeysNext[d.jid] = d.keys;
      setRawKeysByJob(rawKeysNext);

      // Phase 2: fetch every legacy key for each job in parallel.
      const fetchTasks: Array<Promise<{ jid: string; key: string; points: LossPoint[] }>> = [];
      for (const { jid, keys } of discovery) {
        const keep = keys.filter(k => !k.startsWith('_meta/') && k !== 'learning_rate');
        if (!lastStepByJobKeyRef.current[jid]) lastStepByJobKeyRef.current[jid] = {};
        for (const k of keep) {
          const params: Record<string, any> = { key: k };
          const lastStep = lastStepByJobKeyRef.current[jid]?.[k];
          if (reloadInterval && lastStep != null) params.since_step = lastStep;
          params.limit = 1000000;
          fetchTasks.push(
            apiClient
              .get(`/api/jobs/${jid}/loss`, { params })
              .then(res => ({ jid, key: k, points: (res.data?.points ?? []) as LossPoint[] }))
              .catch(() => ({ jid, key: k, points: [] as LossPoint[] })),
          );
        }
      }

      const fetched = await Promise.all(fetchTasks);

      // Drop stale results if our generation was bumped during Phase 2.
      if (myGen !== generationRef.current) return;

      setSeriesByJob(prev => {
        const next: Record<string, SeriesMap> = {};
        // start from existing per-job maps so we keep historical points across
        // incremental polls.
        for (const jid of currentJobIDs) next[jid] = { ...(prev[jid] ?? {}) };

        // group by (job, canonical key)
        const grouped = new Map<string, LossPoint[]>();
        for (const r of fetched) {
          // Skip results for jobs no longer in the active set (defensive
          // guard; the generation check above handles this in practice).
          if (!(r.jid in next)) continue;
          const canonical = canonicalizeKey(r.key);
          const k = `${r.jid}::${canonical}`;
          const list = grouped.get(k) ?? [];
          for (const p of r.points) if (p.value !== null) list.push(p);
          grouped.set(k, list);
          if (r.points.length) {
            const lastStep = r.points[r.points.length - 1].step;
            if (!lastStepByJobKeyRef.current[r.jid]) lastStepByJobKeyRef.current[r.jid] = {};
            const prevLast = lastStepByJobKeyRef.current[r.jid][r.key];
            if (prevLast == null || lastStep > prevLast) {
              lastStepByJobKeyRef.current[r.jid][r.key] = lastStep;
            }
          }
        }

        for (const [groupKey, pts] of grouped.entries()) {
          const sep = groupKey.indexOf('::');
          const jid = groupKey.slice(0, sep);
          const canonical = groupKey.slice(sep + 2);
          if (!next[jid]) continue;
          const existing = next[jid][canonical] ?? [];
          const stepMap = new Map<number, LossPoint>();
          for (const p of existing) stepMap.set(p.step, p);
          for (const p of pts) stepMap.set(p.step, p);
          next[jid][canonical] = Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
        }

        return next;
      });

      // garbage collect lastStepByJobKey entries for removed jobs
      for (const jid of Object.keys(lastStepByJobKeyRef.current)) {
        if (!currentJobIDs.includes(jid)) delete lastStepByJobKeyRef.current[jid];
      }

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching multi-job metrics:', err);
      // Only flag error if a fresher refresh hasn't already taken over.
      if (myGen === generationRef.current) setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [reloadInterval]);

  useEffect(() => {
    // Generation was already bumped synchronously during render (see the
    // latestJobIDsRef sync above). Reset per-job state and kick off a fresh
    // refresh. We also release the inFlight guard: any in-progress refresh
    // is now stale (it'll discard its results via the gen check) and we
    // want the new wave to start immediately, not wait on it.
    inFlightRef.current = false;
    didInitialLoadRef.current = false;
    lastStepByJobKeyRef.current = {};
    setSeriesByJob({});
    setRawKeysByJob({});
    setStatus('idle');

    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobIDsKey, reloadInterval, refresh]);

  return {
    seriesByJob,
    canonicalKeysByJob,
    unionCanonicalKeys,
    intersectCanonicalKeys,
    status,
    refresh,
  };
}
