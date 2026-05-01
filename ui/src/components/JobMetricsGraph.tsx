'use client';

import { Job } from '@prisma/client';
import {
  useJobMetricsLog,
  LossPoint,
  LossBreakdown,
  subsystemOf,
} from '@/hooks/useJobLossLog';
import { useMemo, useState, useEffect } from 'react';
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
// New "Metrics" tab — parallel to the legacy `JobLossGraph` and
// designed to surface the canonical `subsystem/kind/variant` namespace
// introduced in step 4. Keeps feature parity with the old graph for
// smoothing / log-Y / stride / window, and adds:
//   - View dropdown that filters by subsystem (Identity, Body shape, ...)
//   - Facet dropdown: none | by t-band | by sample
//   - Custom tooltip rendering the per-sample breakdown when present.
// =====================================================================

interface Props {
  job: Job;
}

// View / subsystem dropdown options. The first segment of the canonical
// key matches one of these. `Custom` catches anything unmapped.
const VIEW_OPTIONS = [
  { value: 'overview', label: 'Overview' },
  { value: 'core', label: 'Core' },
  { value: 'identity', label: 'Identity' },
  { value: 'body_shape', label: 'Body shape' },
  { value: 'body_proportion', label: 'Body proportion' },
  { value: 'landmark', label: 'Landmark' },
  { value: 'depth', label: 'Depth' },
  { value: 'vae_anchor', label: 'VAE anchor' },
  { value: 'normal', label: 'Normals' },
  { value: 'tokens', label: 'Tokens' },
  { value: 'diffusion', label: 'Diffusion' },
  { value: 'aux', label: 'Aux' },
  { value: 'custom', label: 'Custom' },
] as const;

type ViewKey = (typeof VIEW_OPTIONS)[number]['value'];
type FacetKey = 'none' | 'by_t_band' | 'by_sample';

const PALETTE = [
  'rgba(96,165,250,1)',
  'rgba(52,211,153,1)',
  'rgba(167,139,250,1)',
  'rgba(251,191,36,1)',
  'rgba(244,114,182,1)',
  'rgba(248,113,113,1)',
  'rgba(34,211,238,1)',
  'rgba(129,140,248,1)',
];

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
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

// Detect t-band variants (e.g. `identity/sim/t40`). Returns the parent
// metric name (`identity/sim`) and the band suffix (`t40`) when matched.
const T_BAND_PATTERN = /^(.+?)\/t(\d{2,3})$/;
function splitTBand(canonical: string): { parent: string; band: string } | null {
  const m = T_BAND_PATTERN.exec(canonical);
  if (!m) return null;
  return { parent: m[1], band: `t${m[2]}` };
}

// Group a list of canonical keys for the "by t-band" facet: every parent
// metric maps to the list of its t-band siblings.
function buildTBandGroups(keys: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const k of keys) {
    const sp = splitTBand(k);
    if (!sp) continue;
    (groups[sp.parent] ||= []).push(k);
  }
  for (const k of Object.keys(groups)) groups[k].sort();
  return groups;
}

// Synthetic series-key separator for the "by sample" facet. We encode the
// per-sample synthetic keys as `${metricKey}::sample::${sampleName}` so they
// never collide with real canonical keys (canonical keys never contain `::`).
const SAMPLE_PIVOT_SEP = '::sample::';

function makeSampleKey(metricKey: string, sample: string) {
  return `${metricKey}${SAMPLE_PIVOT_SEP}${sample}`;
}

function splitSampleKey(syntheticKey: string): { metric: string; sample: string } | null {
  const idx = syntheticKey.indexOf(SAMPLE_PIVOT_SEP);
  if (idx === -1) return null;
  return {
    metric: syntheticKey.slice(0, idx),
    sample: syntheticKey.slice(idx + SAMPLE_PIVOT_SEP.length),
  };
}

// Default cap for the number of per-sample series rendered at once.
const BY_SAMPLE_DEFAULT_TOP_N = 12;

interface PivotedSample {
  sample: string;
  count: number;
  points: { step: number; value: number; t?: number }[];
}

// Pivot a series of LossPoints with breakdowns into one synthetic series per
// distinct sample name. Each synthetic series only carries points for the
// steps where that sample appeared in the breakdown payload (sparse).
function pivotBySample(points: LossPoint[]): PivotedSample[] {
  const bySample = new Map<string, PivotedSample>();
  for (const p of points) {
    const bd = p.breakdown;
    if (!bd || !Array.isArray(bd.samples)) continue;
    for (const s of bd.samples) {
      const tag = (s.sample ?? '').trim();
      if (!tag) continue;
      if (!Number.isFinite(s.value)) continue;
      let entry = bySample.get(tag);
      if (!entry) {
        entry = { sample: tag, count: 0, points: [] };
        bySample.set(tag, entry);
      }
      entry.points.push({ step: p.step, value: s.value, t: s.t });
      entry.count += 1;
    }
  }
  // Sort each series's points by step (defensive — points should already be
  // ordered by step from the API but breakdowns are unordered within a step).
  for (const entry of bySample.values()) {
    entry.points.sort((a, b) => a.step - b.step);
  }
  // Rank by frequency (most-frequent samples first), then alphabetically as
  // a stable tie-breaker.
  return Array.from(bySample.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.sample.localeCompare(b.sample);
  });
}

export default function JobMetricsGraph({ job }: Props) {
  const { series, canonicalKeys, status, refresh } = useJobMetricsLog(job.id, 2000);

  const [view, setView] = useState<ViewKey>('overview');
  const [facet, setFacet] = useState<FacetKey>('none');
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showTrend, setShowTrend] = useState(true);
  const [smoothing, setSmoothing] = useState(90);
  const [plotStride, setPlotStride] = useState(1);
  const [windowSize, setWindowSize] = useState<number>(0);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  // by_sample-facet-only state: which metric to fan into per-sample series,
  // and whether to show all sample series or just the top-N most-frequent.
  const [bySampleMetric, setBySampleMetric] = useState<string | null>(null);
  const [bySampleShowAll, setBySampleShowAll] = useState(false);

  // Group keys by subsystem for the view filter.
  const subsystems = useMemo(() => {
    const set = new Set<string>();
    for (const k of canonicalKeys) set.add(subsystemOf(k));
    return Array.from(set).sort();
  }, [canonicalKeys]);

  // The keys visible under the current view (no facet applied yet).
  const viewKeys = useMemo(() => {
    if (view === 'overview') return canonicalKeys;
    if (view === 'custom') {
      // "Custom" = any subsystem not covered by the named views.
      const named = new Set<string>(
        VIEW_OPTIONS.map(v => v.value).filter(v => v !== 'overview' && v !== 'custom'),
      );
      return canonicalKeys.filter(k => !named.has(subsystemOf(k)));
    }
    return canonicalKeys.filter(k => subsystemOf(k) === view);
  }, [view, canonicalKeys]);

  // Candidate metric keys for the "by sample" facet — i.e. metrics in the
  // current view that have at least one point with a per-sample breakdown.
  // Used both to decide whether by_sample is even meaningful and to populate
  // the "active metric" dropdown that appears when by_sample is on.
  const bySampleCandidates = useMemo(
    () =>
      viewKeys.filter(k => {
        const pts: LossPoint[] = series[k] ?? [];
        return pts.some(p => p.breakdown != null);
      }),
    [viewKeys, series],
  );

  // Resolve which metric to fan when by_sample is active. Preserve the
  // user's pick across re-renders, but if the previously-selected metric
  // is no longer a candidate (e.g. they changed view), fall back to the
  // first candidate.
  const activeBySampleMetric = useMemo<string | null>(() => {
    if (facet !== 'by_sample') return null;
    if (bySampleMetric && bySampleCandidates.includes(bySampleMetric)) {
      return bySampleMetric;
    }
    return bySampleCandidates[0] ?? null;
  }, [facet, bySampleMetric, bySampleCandidates]);

  // Pivot the active by_sample metric into synthetic per-sample series. We
  // produce them once here (rather than only inside `perSeries`) so that the
  // toggle list and the pivoted points share the same source of truth.
  const pivotedSamples = useMemo<PivotedSample[]>(() => {
    if (facet !== 'by_sample' || !activeBySampleMetric) return [];
    return pivotBySample(series[activeBySampleMetric] ?? []);
  }, [facet, activeBySampleMetric, series]);

  // Compute the active series after the facet step. The "by t-band"
  // facet keeps every t-band variant; "by sample" replaces the metric
  // key list with synthetic `${metric}::sample::${sample}` keys.
  const facetedKeys = useMemo(() => {
    if (facet === 'none') {
      // Hide t-band siblings under "none" so a metric like `identity/sim`
      // doesn't double-count its own bands. The bands stay accessible
      // via the "by t-band" facet.
      return viewKeys.filter(k => splitTBand(k) === null);
    }
    if (facet === 'by_t_band') {
      const groups = buildTBandGroups(viewKeys);
      return Object.values(groups).flat();
    }
    // by_sample: synthetic per-sample series for the active metric. Cap to
    // top-N most-frequent samples by default to keep the legend / chart
    // readable on large datasets.
    if (!activeBySampleMetric) return [];
    const ranked = pivotedSamples;
    const capped = bySampleShowAll ? ranked : ranked.slice(0, BY_SAMPLE_DEFAULT_TOP_N);
    return capped.map(p => makeSampleKey(activeBySampleMetric, p.sample));
  }, [facet, viewKeys, activeBySampleMetric, pivotedSamples, bySampleShowAll]);

  // Default ON for the current view's keys so users see something on tab
  // open without needing to toggle.
  useEffect(() => {
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of facetedKeys) {
        if (next[k] === undefined) next[k] = true;
      }
      // Clear toggles for keys no longer in scope.
      for (const k of Object.keys(next)) {
        if (!facetedKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [facetedKeys]);

  const activeKeys = useMemo(
    () => facetedKeys.filter(k => enabled[k] !== false),
    [facetedKeys, enabled],
  );

  // Build the chart-friendly data array (raw + EMA-smoothed series merged
  // by step). Mirrors the legacy graph's pipeline to keep the look-and-feel
  // consistent.
  //
  // Two source modes:
  //   - normal: pull raw values from `series[key]` (one point per step).
  //   - by_sample: each `key` is a synthetic `${metric}::sample::${sample}`,
  //     and its raw points come from the pivoted breakdown samples (sparse).
  //     EMA smoothing is suppressed in this mode because per-sample series
  //     are non-contiguous (a sample only appears in steps where it was a
  //     top-K outlier), and applying EMA across those gaps would imply a
  //     continuity that isn't real.
  const perSeries = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98;

    const out: Record<
      string,
      {
        raw: { step: number; value: number; t?: number; breakdown?: LossBreakdown }[];
        smooth: { step: number; value: number }[];
      }
    > = {};

    // Index pivoted samples by name for cheap lookups in by_sample mode.
    const pivotedBySample = new Map<string, PivotedSample>();
    if (facet === 'by_sample') {
      for (const p of pivotedSamples) pivotedBySample.set(p.sample, p);
    }

    for (const key of activeKeys) {
      let raw: { step: number; value: number; t?: number; breakdown?: LossBreakdown }[] = [];

      if (facet === 'by_sample') {
        const split = splitSampleKey(key);
        if (!split) continue;
        const entry = pivotedBySample.get(split.sample);
        if (!entry) continue;
        raw = entry.points
          .filter(p => Number.isFinite(p.value))
          .filter(p => (useLogScale ? p.value > 0 : true))
          .filter((_, idx) => idx % stride === 0)
          .map(p => ({ step: p.step, value: p.value, t: p.t }));
      } else {
        const pts: LossPoint[] = series[key] ?? [];
        raw = pts
          .filter(p => p.value !== null && Number.isFinite(p.value as number))
          .map(p => ({
            step: p.step,
            value: p.value as number,
            breakdown: p.breakdown,
          }))
          .filter(p => (useLogScale ? p.value > 0 : true))
          .filter((_, idx) => idx % stride === 0);
      }

      if (windowSize > 0 && raw.length > windowSize) {
        raw = raw.slice(raw.length - windowSize);
      }
      // EMA across non-contiguous per-sample series is misleading — disable.
      const smooth =
        facet === 'by_sample'
          ? []
          : emaSmoothPoints(
              raw.map(r => ({ step: r.step, value: r.value })),
              alpha,
            );
      out[key] = { raw, smooth };
    }

    return out;
  }, [
    series,
    activeKeys,
    smoothing,
    plotStride,
    windowSize,
    useLogScale,
    facet,
    pivotedSamples,
  ]);

  // Lookup of breakdown payloads by `${key}@${step}` so the tooltip can
  // render top-K samples without re-walking every point.
  const breakdownByKeyStep = useMemo(() => {
    const map = new Map<string, LossBreakdown>();
    for (const key of activeKeys) {
      const pts = perSeries[key]?.raw ?? [];
      for (const p of pts) {
        if (p.breakdown) {
          map.set(`${key}@${p.step}`, p.breakdown);
        }
      }
    }
    return map;
  }, [activeKeys, perSeries]);

  // Lookup of per-sample t values by `${syntheticKey}@${step}` so the
  // tooltip in by_sample mode can show the diffusion-t at which a sample
  // was sampled, in addition to the metric value.
  const sampleTByKeyStep = useMemo(() => {
    const map = new Map<string, number>();
    if (facet !== 'by_sample') return map;
    for (const key of activeKeys) {
      const pts = perSeries[key]?.raw ?? [];
      for (const p of pts) {
        if (typeof p.t === 'number' && Number.isFinite(p.t)) {
          map.set(`${key}@${p.step}`, p.t);
        }
      }
    }
    return map;
  }, [facet, activeKeys, perSeries]);

  // Per-sample linear-regression trend lines (by_sample only). With ≥ 2 raw
  // points we fit y = m*step + b by least squares; we only need the endpoints
  // (min/max step) because Recharts draws a straight line between them.
  // Skipped on log Y when an endpoint isn't strictly positive.
  const trends = useMemo(() => {
    const out: Record<
      string,
      { yMin: number; yMax: number; minStep: number; maxStep: number } | null
    > = {};
    if (facet !== 'by_sample') return out;
    for (const key of activeKeys) {
      const pts = perSeries[key]?.raw ?? [];
      if (pts.length < 2) {
        out[key] = null;
        continue;
      }
      let n = 0,
        sx = 0,
        sy = 0,
        sxy = 0,
        sxx = 0;
      let minStep = Infinity,
        maxStep = -Infinity;
      for (const p of pts) {
        n++;
        sx += p.step;
        sy += p.value;
        sxy += p.step * p.value;
        sxx += p.step * p.step;
        if (p.step < minStep) minStep = p.step;
        if (p.step > maxStep) maxStep = p.step;
      }
      const denom = n * sxx - sx * sx;
      let m: number;
      let b: number;
      if (denom === 0) {
        m = 0;
        b = sy / n;
      } else {
        m = (n * sxy - sx * sy) / denom;
        b = (sy - m * sx) / n;
      }
      const yMin = m * minStep + b;
      const yMax = m * maxStep + b;
      if (useLogScale && (yMin <= 0 || yMax <= 0)) {
        out[key] = null;
        continue;
      }
      out[key] = { yMin, yMax, minStep, maxStep };
    }
    return out;
  }, [facet, activeKeys, perSeries, useLogScale]);

  const chartData = useMemo(() => {
    const m = new Map<number, any>();
    for (const key of activeKeys) {
      const s = perSeries[key];
      if (!s) continue;
      for (const p of s.raw) {
        const row = m.get(p.step) ?? { step: p.step };
        row[`${key}__raw`] = p.value;
        m.set(p.step, row);
      }
      for (const p of s.smooth) {
        const row = m.get(p.step) ?? { step: p.step };
        row[`${key}__smooth`] = p.value;
        m.set(p.step, row);
      }
    }
    // Inject trend endpoints (by_sample only). Two rows per series — Recharts
    // with connectNulls draws a straight line between them.
    for (const key of activeKeys) {
      const t = trends[key];
      if (!t) continue;
      const rowMin = m.get(t.minStep) ?? { step: t.minStep };
      rowMin[`${key}__trend`] = t.yMin;
      m.set(t.minStep, rowMin);
      const rowMax = m.get(t.maxStep) ?? { step: t.maxStep };
      rowMax[`${key}__trend`] = t.yMax;
      m.set(t.maxStep, rowMax);
    }
    return Array.from(m.values()).sort((a, b) => a.step - b.step);
  }, [activeKeys, perSeries, trends]);

  const hasData = chartData.length > 1;

  // Custom tooltip: shows scalar + (when available) the per-sample
  // breakdown's mean / n / std and the top-K samples by deviation.
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
          const seriesKey = dataKey.endsWith('__smooth')
            ? dataKey.slice(0, -'__smooth'.length)
            : dataKey.endsWith('__raw')
              ? dataKey.slice(0, -'__raw'.length)
              : dataKey;
          // by_sample synthetic keys: show only the sample name + t.
          const sampleSplit = splitSampleKey(seriesKey);
          if (sampleSplit) {
            const tval = sampleTByKeyStep.get(`${seriesKey}@${label}`);
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
                    {sampleSplit.sample}
                  </span>
                  <span style={{ marginLeft: 'auto', color: 'rgba(255,255,255,0.85)' }}>
                    {formatNum(Number(p.value))}
                  </span>
                </div>
                <div style={{ marginLeft: 14, marginTop: 2, color: 'rgba(255,255,255,0.55)', fontSize: 11 }}>
                  {sampleSplit.metric}
                  {tval != null ? ` · t=${tval.toFixed(2)}` : ''}
                </div>
              </div>
            );
          }
          const breakdown = breakdownByKeyStep.get(`${seriesKey}@${label}`);
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
                <span>{p.name}</span>
                <span style={{ marginLeft: 'auto', color: 'rgba(255,255,255,0.85)' }}>
                  {formatNum(Number(p.value))}
                </span>
              </div>
              {breakdown && (
                <div style={{ marginLeft: 14, marginTop: 4, color: 'rgba(255,255,255,0.65)' }}>
                  <div>
                    n={breakdown.n}
                    {breakdown.mean != null ? ` · mean=${formatNum(breakdown.mean)}` : ''}
                    {breakdown.std != null ? ` · std=${formatNum(breakdown.std)}` : ''}
                  </div>
                  {breakdown.samples?.length ? (
                    <ul style={{ margin: '4px 0 0 0', padding: 0, listStyle: 'none' }}>
                      {breakdown.samples.slice(0, 8).map((s, j) => (
                        <li
                          key={j}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 8,
                            color: 'rgba(255,255,255,0.7)',
                          }}
                        >
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {s.sample ?? '(unknown)'}
                            {s.t != null ? ` · t=${s.t.toFixed(2)}` : ''}
                          </span>
                          <span>{formatNum(s.value)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              )}
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
          <div className="h-2 w-2 rounded-full bg-emerald-400" />
          <h2 className="text-gray-100 text-sm font-medium">Metrics (new)</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${chartData.length.toLocaleString()} steps · ${activeKeys.length} series`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={view}
            onChange={e => setView(e.target.value as ViewKey)}
            className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200"
          >
            {VIEW_OPTIONS.map(v => {
              const count = canonicalKeys.filter(k =>
                v.value === 'overview'
                  ? true
                  : v.value === 'custom'
                    ? !VIEW_OPTIONS.some(o => o.value !== 'overview' && o.value !== 'custom' && o.value === subsystemOf(k))
                    : subsystemOf(k) === v.value,
              ).length;
              return (
                <option key={v.value} value={v.value}>
                  {v.label}
                  {count ? ` (${count})` : ''}
                </option>
              );
            })}
          </select>
          <select
            value={facet}
            onChange={e => setFacet(e.target.value as FacetKey)}
            className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200"
            title="By sample: fan a metric into one line per training-image filename, using the per-sample breakdown payload."
          >
            <option value="none">Facet: none</option>
            <option value="by_t_band">Facet: by t-band</option>
            <option value="by_sample">Facet: by sample</option>
          </select>
          {facet === 'by_sample' && bySampleCandidates.length > 0 && (
            <select
              value={activeBySampleMetric ?? ''}
              onChange={e => setBySampleMetric(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200"
              title="Metric to fan into per-sample series."
            >
              {bySampleCandidates.map(k => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={refresh}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* by_sample mode hint: clarifies the sparseness / outlier-bias of
          the underlying breakdown payload so users don't read too much
          into a "missing" sample-step combination. */}
      {facet === 'by_sample' && (
        <div className="px-4 pt-3 -mb-1 text-[11px] text-gray-500">
          {bySampleCandidates.length === 0
            ? 'No metrics in this view carry per-sample breakdowns yet. Run with the new MetricBuffer enabled to populate this facet.'
            : `Showing per-sample values for ${activeBySampleMetric ?? '(none)'} · dots are observations${showTrend ? ', dashed lines are linear-regression trends' : ''}. ` +
              'Each step retains up to 16 samples per metric (top-K by deviation from the batch mean), so samples that are perfectly average may not appear at every step.'}
        </div>
      )}

      {/* Chart */}
      <div className="px-4 pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative">
          {!hasData ? (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              {status === 'error' ? 'Failed to load metrics.' : 'Waiting for points...'}
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

                {activeKeys.map(k => {
                  const color = strokeForKey(k);
                  // by_sample: draw dots only (no connecting line) because
                  // per-sample series are sparse and a connecting line would
                  // imply continuity between non-adjacent observations.
                  if (facet === 'by_sample') {
                    const split = splitSampleKey(k);
                    const displayName = split ? split.sample : k;
                    const sampleColor = split ? strokeForKey(split.sample) : color;
                    return (
                      <g key={k}>
                        {showTrend && trends[k] && (
                          <Line
                            type="linear"
                            dataKey={`${k}__trend`}
                            stroke={sampleColor}
                            strokeWidth={1.25}
                            strokeDasharray="4 3"
                            strokeOpacity={0.75}
                            dot={false}
                            activeDot={false}
                            isAnimationActive={false}
                            connectNulls
                            legendType="none"
                          />
                        )}
                        <Line
                          type="monotone"
                          dataKey={`${k}__raw`}
                          name={displayName}
                          stroke={sampleColor}
                          strokeWidth={0}
                          dot={{ r: 3, stroke: sampleColor, strokeWidth: 1, fill: sampleColor }}
                          activeDot={{ r: 5 }}
                          isAnimationActive={false}
                        />
                      </g>
                    );
                  }
                  return (
                    <g key={k}>
                      {showRaw && (
                        <Line
                          type="monotone"
                          dataKey={`${k}__raw`}
                          name={`${k} (raw)`}
                          stroke={color.replace('1)', '0.40)')}
                          strokeWidth={1.25}
                          dot={false}
                          isAnimationActive={false}
                          connectNulls
                        />
                      )}
                      {showSmoothed && (
                        <Line
                          type="monotone"
                          dataKey={`${k}__smooth`}
                          name={k}
                          stroke={color}
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                          connectNulls
                        />
                      )}
                    </g>
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 pb-2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
              {facet === 'by_sample' && (
                <ToggleButton checked={showTrend} onClick={() => setShowTrend(v => !v)} label="Trend" />
              )}
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-xs text-gray-400">
                {facet === 'by_sample' ? `Samples (${facetedKeys.length})` : `Series (${facetedKeys.length})`}
              </label>
              {facet === 'by_sample' && pivotedSamples.length > BY_SAMPLE_DEFAULT_TOP_N && (
                <button
                  type="button"
                  onClick={() => setBySampleShowAll(v => !v)}
                  className="text-[11px] px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700"
                >
                  {bySampleShowAll
                    ? `Top ${BY_SAMPLE_DEFAULT_TOP_N}`
                    : `Show all (${pivotedSamples.length})`}
                </button>
              )}
            </div>
            {facetedKeys.length === 0 ? (
              <div className="text-sm text-gray-400">
                {facet === 'by_sample'
                  ? 'No metrics with per-sample breakdowns in this view.'
                  : 'No metrics in this view.'}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2 max-h-32 overflow-auto">
                {facetedKeys.map(k => {
                  const split = splitSampleKey(k);
                  const display = split ? split.sample : k;
                  const swatch = split ? strokeForKey(split.sample) : strokeForKey(k);
                  return (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setEnabled(prev => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                      className={[
                        'px-2 py-1 rounded text-[11px] border transition-colors',
                        enabled[k] === false
                          ? 'bg-gray-900 text-gray-500 border-gray-800 hover:bg-gray-800/60'
                          : 'bg-gray-900 text-gray-200 border-gray-800 hover:bg-gray-800/60',
                      ].join(' ')}
                      aria-pressed={enabled[k] !== false}
                      title={k}
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-full mr-1.5"
                        style={{ background: swatch }}
                      />
                      {display}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">
                Smoothing
                {facet === 'by_sample' && (
                  <span className="ml-2 text-[10px] text-gray-500">(disabled in by-sample mode)</span>
                )}
              </label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-emerald-500"
              disabled={!showSmoothed || facet === 'by_sample'}
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
              className="w-full accent-emerald-500"
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
              className="w-full accent-emerald-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">
              Set to 0 to show all (not recommended for very long runs).
            </div>
          </div>
        </div>

        {/* Subsystem summary chip row — quick visual confirmation that
            the dual-write covered every expected namespace. */}
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-gray-500">
          <span>subsystems present:</span>
          {subsystems.map(s => (
            <span key={s} className="px-2 py-0.5 rounded bg-gray-800 text-gray-300">
              {s}
            </span>
          ))}
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
          ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
