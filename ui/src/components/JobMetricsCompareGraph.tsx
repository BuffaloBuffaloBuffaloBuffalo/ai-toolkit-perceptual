'use client';

import { Job } from '@prisma/client';
import {
  useMultiJobMetricsLog,
  LossPoint,
  subsystemOf,
} from '@/hooks/useJobLossLog';
import useJobsList from '@/hooks/useJobsList';
import { useEffect, useMemo, useState } from 'react';
import useLocalStorageState from '@/hooks/useLocalStorageState';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';

// =====================================================================
// Cross-job metrics comparison.
//
// Renders a single canonical metric series for each of N selected jobs,
// distinguished by color. Built to live alongside the per-job metrics
// graph; reuses the same fetching pipeline (canonical keys + per-step
// points) but fans across many jobs.
//
// UX:
//   - Anchor job is whatever page you came from. Always preselected.
//   - "Add job" multi-picker (a checklist of all jobs in the DB).
//   - Single metric dropdown (the canonical key being compared).
//     Defaults to a sensible loss key shared across jobs.
//   - Standard smoothing / log-Y / stride / window controls.
// =====================================================================

interface Props {
  job: Job;
}

const JOB_PALETTE = [
  // Hand-tuned for distinguishability against the dark background.
  'rgba(96,165,250,1)', // blue-400
  'rgba(248,113,113,1)', // red-400
  'rgba(52,211,153,1)', // emerald-400
  'rgba(251,191,36,1)', // amber-400
  'rgba(167,139,250,1)', // purple-400
  'rgba(244,114,182,1)', // pink-400
  'rgba(34,211,238,1)', // cyan-400
  'rgba(129,140,248,1)', // indigo-400
  'rgba(250,204,21,1)', // yellow-400
  'rgba(74,222,128,1)', // green-400
  'rgba(252,165,165,1)', // red-300
  'rgba(165,180,252,1)', // indigo-300
];

function colorForJob(jobId: string, orderIdx: number) {
  // Stable color by selection order (so the anchor always gets blue, the
  // next pick gets red, etc.). Falls back to a hash if we run out.
  if (orderIdx >= 0 && orderIdx < JOB_PALETTE.length) return JOB_PALETTE[orderIdx];
  let h = 2166136261;
  for (let i = 0; i < jobId.length; i++) {
    h ^= jobId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return JOB_PALETTE[Math.abs(h) % JOB_PALETTE.length];
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function emaSmoothPoints(points: { step: number; value: number }[], alpha: number) {
  if (points.length === 0) return [];
  const a = clamp01(alpha);
  const out: { step: number; value: number }[] = new Array(points.length);
  let prev = points[0].value;
  out[0] = { step: points[0].step, value: prev };
  for (let i = 1; i < points.length; i++) {
    const x = points[i].value;
    prev = a * x + (1 - a) * prev;
    out[i] = { step: points[i].step, value: prev };
  }
  return out;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

// Pick a sensible default metric to compare. Prefer canonical loss-style
// keys; fall back to whatever's available.
function pickDefaultMetric(candidates: string[]): string | null {
  if (!candidates.length) return null;
  const priority = [
    'core/loss',
    'diffusion/loss_applied',
    'identity/loss_applied',
    'depth/loss_applied',
    'body_shape/loss_applied',
    'normal/loss_applied',
    'body_proportion/loss_applied',
    'identity/sim',
  ];
  for (const p of priority) if (candidates.includes(p)) return p;
  // First "loss" anything wins next.
  const loss = candidates.find(k => k.toLowerCase().includes('loss'));
  if (loss) return loss;
  return candidates[0];
}

export default function JobMetricsCompareGraph({ job }: Props) {
  const { jobs: allJobs, status: jobsStatus } = useJobsList(false, 30000);

  // Per-job prefs persist in localStorage. Anchor job (`job`) is always at
  // index 0 of `selectedJobIDs`; user-toggled additions are after it.
  const lsKey = (suffix: string) => `aitk:metricsCompare:${job.id}:${suffix}`;
  const [selectedJobIDs, setSelectedJobIDs] = useLocalStorageState<string[]>(lsKey('selectedJobIDs'), [job.id]);

  // Reset when anchor changes (e.g. user navigates to a different job).
  useEffect(() => {
    setSelectedJobIDs(prev => (prev[0] === job.id ? prev : [job.id]));
  }, [job.id, setSelectedJobIDs]);

  // 8s reload interval is intentional: an N-job, M-key fetch can issue
  // hundreds of requests per refresh, and the browser's per-origin
  // connection limit means short intervals stack faster than they drain.
  const { seriesByJob, unionCanonicalKeys, intersectCanonicalKeys, status, refresh } =
    useMultiJobMetricsLog(selectedJobIDs, 8000);

  // Subsystem filter: lets the user narrow the metric dropdown when many
  // canonical keys are present. Defaults to "all".
  const [subsystem, setSubsystem] = useLocalStorageState<string>(lsKey('subsystem'), 'all');
  const [keyMode, setKeyMode] = useLocalStorageState<'union' | 'intersect'>(lsKey('keyMode'), 'intersect');
  const [metric, setMetric] = useLocalStorageState<string | null>(lsKey('metric'), null);

  // Smoothing / display controls (mirrors the single-job graph).
  const [useLogScale, setUseLogScale] = useLocalStorageState(lsKey('useLogScale'), false);
  const [showRaw, setShowRaw] = useLocalStorageState(lsKey('showRaw'), false);
  const [showSmoothed, setShowSmoothed] = useLocalStorageState(lsKey('showSmoothed'), true);
  const [smoothing, setSmoothing] = useLocalStorageState(lsKey('smoothing'), 0);
  const [plotStride, setPlotStride] = useLocalStorageState(lsKey('plotStride'), 1);
  const [windowSize, setWindowSize] = useLocalStorageState<number>(lsKey('windowSize'), 0);

  const subsystems = useMemo(() => {
    const set = new Set<string>();
    for (const k of unionCanonicalKeys) set.add(subsystemOf(k));
    return ['all', ...Array.from(set).sort()];
  }, [unionCanonicalKeys]);

  const metricCandidates = useMemo(() => {
    const base = keyMode === 'intersect' ? intersectCanonicalKeys : unionCanonicalKeys;
    if (subsystem === 'all') return base;
    return base.filter(k => subsystemOf(k) === subsystem);
  }, [keyMode, intersectCanonicalKeys, unionCanonicalKeys, subsystem]);

  // Keep `metric` valid when candidates change. If the current pick
  // disappears from candidates, fall back to a default.
  useEffect(() => {
    if (metric && metricCandidates.includes(metric)) return;
    setMetric(pickDefaultMetric(metricCandidates));
  }, [metric, metricCandidates]);

  const jobsById = useMemo(() => {
    const m = new Map<string, Job>();
    for (const j of allJobs) m.set(j.id, j);
    // ensure anchor is included even if it hasn't shown up in /api/jobs yet
    m.set(job.id, job);
    return m;
  }, [allJobs, job]);

  // Per-job processed series for the active metric.
  const perJobSeries = useMemo(() => {
    if (!metric) return {} as Record<string, { raw: { step: number; value: number }[]; smooth: { step: number; value: number }[] }>;
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98;

    const out: Record<string, { raw: { step: number; value: number }[]; smooth: { step: number; value: number }[] }> = {};
    for (const jid of selectedJobIDs) {
      const pts: LossPoint[] = (seriesByJob[jid] ?? {})[metric] ?? [];
      let raw = pts
        .filter(p => p.value !== null && Number.isFinite(p.value as number))
        .map(p => ({ step: p.step, value: p.value as number }))
        .filter(p => (useLogScale ? p.value > 0 : true))
        .filter((_, idx) => idx % stride === 0);
      if (windowSize > 0 && raw.length > windowSize) {
        raw = raw.slice(raw.length - windowSize);
      }
      const smooth = emaSmoothPoints(raw, alpha);
      out[jid] = { raw, smooth };
    }
    return out;
  }, [metric, selectedJobIDs, seriesByJob, smoothing, plotStride, windowSize, useLogScale]);

  // Merge into one Recharts-friendly array, indexed by step. Each job
  // gets `${jid}__raw` and `${jid}__smooth` columns.
  const chartData = useMemo(() => {
    const m = new Map<number, any>();
    for (const jid of selectedJobIDs) {
      const s = perJobSeries[jid];
      if (!s) continue;
      for (const p of s.raw) {
        const row = m.get(p.step) ?? { step: p.step };
        row[`${jid}__raw`] = p.value;
        m.set(p.step, row);
      }
      for (const p of s.smooth) {
        const row = m.get(p.step) ?? { step: p.step };
        row[`${jid}__smooth`] = p.value;
        m.set(p.step, row);
      }
    }
    return Array.from(m.values()).sort((a, b) => a.step - b.step);
  }, [selectedJobIDs, perJobSeries]);

  const hasData = chartData.length > 1;

  function toggleJob(jid: string) {
    setSelectedJobIDs(prev => {
      if (prev.includes(jid)) {
        // Don't allow removing the anchor — it's always pinned.
        if (jid === job.id) return prev;
        return prev.filter(x => x !== jid);
      }
      return [...prev, jid];
    });
  }

  // Sort jobs in the picker: running first, then by created_at desc (which
  // is what /api/jobs already returns).
  const pickerJobs = useMemo(() => {
    const live = ['running', 'queued', 'stopping'];
    return [...allJobs].sort((a, b) => {
      const aLive = live.includes(a.status) ? 0 : 1;
      const bLive = live.includes(b.status) ? 0 : 1;
      if (aLive !== bLive) return aLive - bLive;
      return 0;
    });
  }, [allJobs]);

  // Custom tooltip: shows job name + value per series.
  const renderTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;
    return (
      <div
        style={{
          background: 'rgba(17,24,39,0.96)',
          border: '1px solid rgba(31,41,55,1)',
          borderRadius: 10,
          color: 'rgba(255,255,255,0.92)',
          fontSize: 12,
          padding: 10,
          maxWidth: 360,
        }}
      >
        <div style={{ color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>step {label}</div>
        {payload.map((p: any, i: number) => {
          const dataKey: string = p.dataKey ?? '';
          const jid = dataKey.endsWith('__smooth')
            ? dataKey.slice(0, -'__smooth'.length)
            : dataKey.endsWith('__raw')
              ? dataKey.slice(0, -'__raw'.length)
              : dataKey;
          const j = jobsById.get(jid);
          const name = j?.name ?? jid;
          return (
            <div key={i} style={{ marginTop: i === 0 ? 0 : 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    background: p.color,
                  }}
                />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {name}
                </span>
                <span style={{ marginLeft: 'auto', color: 'rgba(255,255,255,0.85)' }}>
                  {formatNum(Number(p.value))}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-orange-400" />
          <h2 className="text-gray-100 text-sm font-medium">Compare metrics across jobs</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && metric && hasData &&
              `${selectedJobIDs.length} job${selectedJobIDs.length === 1 ? '' : 's'} · ${chartData.length.toLocaleString()} steps`}
            {status === 'success' && (!metric || !hasData) && 'No data yet'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={subsystem}
            onChange={e => setSubsystem(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200"
            title="Filter the metric dropdown by subsystem"
          >
            {subsystems.map(s => (
              <option key={s} value={s}>
                {s === 'all' ? 'All subsystems' : s}
              </option>
            ))}
          </select>
          <select
            value={keyMode}
            onChange={e => setKeyMode(e.target.value as 'union' | 'intersect')}
            className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200"
            title="Intersect: only metrics present in every selected job. Union: any metric present in at least one job."
          >
            <option value="intersect">Keys: intersect</option>
            <option value="union">Keys: union</option>
          </select>
          <select
            value={metric ?? ''}
            onChange={e => setMetric(e.target.value || null)}
            className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200 max-w-[14rem]"
            disabled={metricCandidates.length === 0}
          >
            {metricCandidates.length === 0 ? (
              <option value="">no metrics</option>
            ) : (
              metricCandidates.map(k => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            onClick={refresh}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Selected jobs strip */}
      <div className="px-4 pt-3 -mb-1 flex flex-wrap gap-2 items-center text-[11px]">
        <span className="text-gray-500 mr-1">Comparing:</span>
        {selectedJobIDs.map((jid, idx) => {
          const j = jobsById.get(jid);
          const color = colorForJob(jid, idx);
          return (
            <span
              key={jid}
              className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-gray-800 text-gray-200 border border-gray-700"
              title={jid}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: color }}
              />
              <span className="max-w-[14rem] truncate">{j?.name ?? jid}</span>
              {jid !== job.id && (
                <button
                  type="button"
                  onClick={() => toggleJob(jid)}
                  className="ml-1 text-gray-500 hover:text-gray-200"
                  aria-label={`Remove ${j?.name ?? jid}`}
                >
                  ×
                </button>
              )}
              {jid === job.id && <span className="ml-0.5 text-[10px] text-gray-500">(anchor)</span>}
            </span>
          );
        })}
      </div>

      {/* Chart */}
      <div className="px-4 pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative">
          {!hasData ? (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              {status === 'error'
                ? 'Failed to load metrics.'
                : !metric
                  ? metricCandidates.length === 0 && keyMode === 'intersect' && selectedJobIDs.length > 1
                    ? 'No metrics shared across all selected jobs. Try "Keys: union" or remove a job.'
                    : 'No metrics found.'
                  : 'Waiting for points...'}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 16, bottom: 10, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={40}
                />
                <YAxis
                  scale={useLogScale ? 'log' : 'linear'}
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={72}
                  tickFormatter={formatNum}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={renderTooltip} cursor={{ stroke: 'rgba(59,130,246,0.25)', strokeWidth: 1 }} />
                <Legend wrapperStyle={{ paddingTop: 8, color: 'rgba(255,255,255,0.7)', fontSize: 12 }} />

                {selectedJobIDs.flatMap((jid, idx) => {
                  const color = colorForJob(jid, idx);
                  const name = jobsById.get(jid)?.name ?? jid;
                  const lines: any[] = [];
                  if (showRaw) {
                    lines.push(
                      <Line
                        key={`${jid}__raw`}
                        type="monotone"
                        dataKey={`${jid}__raw`}
                        name={`${name} (raw)`}
                        stroke={color.replace('1)', '0.40)')}
                        strokeWidth={1.25}
                        dot={false}
                        isAnimationActive={false}
                        connectNulls
                      />
                    );
                  }
                  if (showSmoothed) {
                    lines.push(
                      <Line
                        key={`${jid}__smooth`}
                        type="monotone"
                        dataKey={`${jid}__smooth`}
                        name={name}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                        connectNulls
                      />
                    );
                  }
                  return lines;
                })}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 pb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-xs text-gray-400">
                Jobs ({selectedJobIDs.length} of {allJobs.length})
              </label>
              <span className="text-[10px] text-gray-500">
                {jobsStatus === 'loading' ? 'loading…' : ''}
              </span>
            </div>
            <div className="flex flex-wrap gap-2 max-h-32 overflow-auto">
              {pickerJobs.map(j => {
                const checked = selectedJobIDs.includes(j.id);
                const idx = selectedJobIDs.indexOf(j.id);
                const swatch = checked ? colorForJob(j.id, idx) : 'rgba(75,85,99,1)';
                const isAnchor = j.id === job.id;
                return (
                  <button
                    key={j.id}
                    type="button"
                    onClick={() => toggleJob(j.id)}
                    disabled={isAnchor}
                    className={[
                      'px-2 py-1 rounded text-[11px] border transition-colors',
                      checked
                        ? 'bg-gray-900 text-gray-200 border-gray-700 hover:bg-gray-800/60'
                        : 'bg-gray-900 text-gray-500 border-gray-800 hover:bg-gray-800/60',
                      isAnchor ? 'opacity-90 cursor-default' : '',
                    ].join(' ')}
                    aria-pressed={checked}
                    title={isAnchor ? `${j.name} (anchor — always selected)` : j.name}
                  >
                    <span
                      className="inline-block h-2 w-2 rounded-full mr-1.5"
                      style={{ background: swatch }}
                    />
                    {j.name}
                    {isAnchor && <span className="ml-1 text-[10px] text-gray-500">★</span>}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Smoothing</label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-orange-500"
              disabled={!showSmoothed}
            />
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Plot stride</label>
              <span className="text-xs text-gray-300">every {plotStride} pt</span>
            </div>
            <input
              type="range"
              min={1}
              max={20}
              value={plotStride}
              onChange={e => setPlotStride(Number(e.target.value))}
              className="w-full accent-orange-500"
            />
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 md:col-span-2">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Window (last N points)</label>
              <span className="text-xs text-gray-300">{windowSize === 0 ? 'all' : windowSize.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min={0}
              max={20000}
              step={250}
              value={windowSize}
              onChange={e => setWindowSize(Number(e.target.value))}
              className="w-full accent-orange-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">
              Set to 0 to show all (not recommended for very long runs).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-orange-500/10 text-orange-300 border-orange-500/30 hover:bg-orange-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
