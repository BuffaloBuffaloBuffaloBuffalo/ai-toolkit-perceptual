'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import useDepthPreviews, { DepthPreview } from '@/hooks/useDepthPreviews';
import useLocalStorageState from '@/hooks/useLocalStorageState';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { LuImageOff, LuLoader, LuBan, LuX } from 'react-icons/lu';

type SortKey = 'step' | 't' | 'dc';
type SortDir = 'asc' | 'desc';

// Bands match the trainer's convention: bin_start = floor(t * 10) / 10, label
// `t{int(bin_start*100):02d}`. See SDTrainer.py around line 2147 etc.
const BAND_VALUES = ['all', 't00', 't10', 't20', 't30', 't40', 't50', 't60', 't70', 't80', 't90'] as const;
type Band = (typeof BAND_VALUES)[number];
const BAND_SET = new Set<Band>(BAND_VALUES);
const SORT_KEYS = new Set<SortKey>(['step', 't', 'dc']);

function bandFor(t: number): Band {
  const lo = Math.floor(Math.max(0, Math.min(0.999999, t)) * 10) * 10;
  return `t${lo.toString().padStart(2, '0')}` as Band;
}

function basename(p: string): string {
  const i = p.lastIndexOf('/');
  return i >= 0 ? p.slice(i + 1) : p;
}
function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}
function sizeKey(s?: { w: number; h: number }): string | null {
  return s ? `${s.w}x${s.h}` : null;
}

interface Props {
  job: Job;
}

export default function DepthPreviews({ job }: Props) {
  const { previews, status } = useDepthPreviews(job.id, 5000);
  const containerRef = useRef<HTMLDivElement>(null);

  const stepBounds = useMemo<{ lo: number; hi: number }>(() => {
    if (previews.length === 0) return { lo: 0, hi: 0 };
    let lo = Infinity, hi = -Infinity;
    for (const p of previews) {
      if (p.step < lo) lo = p.step;
      if (p.step > hi) hi = p.step;
    }
    return { lo, hi };
  }, [previews]);

  // Filter / sort state. Persisted per-job in localStorage so a refresh,
  // tab-switch-and-return, or browser-restart all land on the last view the
  // user had for this job. Empty strings on the step inputs mean "no
  // constraint" — keeps the controls usable before previews finish loading.
  const lsKey = (suffix: string) => `aitk:depthPreviews:${job.id}:${suffix}`;
  const [minStep, setMinStep] = useLocalStorageState<string>(lsKey('minStep'), '');
  const [maxStep, setMaxStep] = useLocalStorageState<string>(lsKey('maxStep'), '');
  const [minT, setMinT] = useLocalStorageState<number>(lsKey('minT'), 0);
  const [maxT, setMaxT] = useLocalStorageState<number>(lsKey('maxT'), 1);
  const [band, setBand] = useLocalStorageState<Band>(lsKey('band'), 'all');
  const [sample, setSample] = useLocalStorageState<string>(lsKey('sample'), 'all');
  const [size, setSize] = useLocalStorageState<string>(lsKey('size'), 'all');
  const [sortKey, setSortKey] = useLocalStorageState<SortKey>(lsKey('sortKey'), 'step');
  const [sortDir, setSortDir] = useLocalStorageState<SortDir>(lsKey('sortDir'), 'desc');
  // Zoom-overlay state. The full path is transient (it depends on where the
  // training output lives); we persist just the filename and resolve it back
  // to a path once the preview list arrives.
  const [selectedPath, _setSelectedPathTransient] = useState<string | null>(null);
  const [persistedSelectedFile, setPersistedSelectedFile] = useLocalStorageState<string | null>(
    lsKey('selected'),
    null,
  );
  const setSelectedPath = (p: string | null) => {
    _setSelectedPathTransient(p);
    setPersistedSelectedFile(p ? basename(p) : null);
  };
  // Resolve a previously-persisted selection to a full path once previews
  // load. Only runs while no transient selection is active — avoids
  // overwriting an in-progress click with a stale localStorage value.
  useEffect(() => {
    if (selectedPath != null) return;
    if (!persistedSelectedFile) return;
    if (previews.length === 0) return;
    const match = previews.find(p => basename(p.path) === persistedSelectedFile);
    if (match) _setSelectedPathTransient(match.path);
  }, [previews, selectedPath, persistedSelectedFile]);

  // Sample × size options are linked: each dropdown shows only values that
  // *exist in combination with* the other dropdown's current selection. This
  // prevents the user from picking a (sample, size) pair that has no previews.
  // If a setter is called with a value that would leave the partner dropdown
  // on an unreachable value, the partner is reset to "all" — see setters
  // below.
  const sizeSortCmp = (a: string, b: string) => {
    const [aw, ah] = a.split('x').map(n => parseInt(n, 10));
    const [bw, bh] = b.split('x').map(n => parseInt(n, 10));
    return aw - bw || ah - bh;
  };
  const sampleOptions = useMemo(() => {
    const set = new Set<string>();
    for (const p of previews) {
      if (!p.srcName) continue;
      if (size !== 'all' && sizeKey(p.size) !== size) continue;
      set.add(p.srcName);
    }
    return Array.from(set).sort();
  }, [previews, size]);
  const sizeOptions = useMemo(() => {
    const set = new Set<string>();
    for (const p of previews) {
      const k = sizeKey(p.size);
      if (!k) continue;
      if (sample !== 'all' && p.srcName !== sample) continue;
      set.add(k);
    }
    return Array.from(set).sort(sizeSortCmp);
  }, [previews, sample]);

  // When the user picks a value, validate that the other dropdown's current
  // value is still reachable; if not, reset the other to "all" so the UI
  // never displays a stuck-but-unreachable pair.
  const sampleHasSize = (s: string, z: string): boolean => {
    if (z === 'all') return true;
    return previews.some(p => p.srcName === s && sizeKey(p.size) === z);
  };
  const sizeHasSample = (z: string, s: string): boolean => {
    if (s === 'all') return true;
    return previews.some(p => sizeKey(p.size) === z && p.srcName === s);
  };
  const pickSample = (v: string) => {
    setSample(v);
    if (v !== 'all' && size !== 'all' && !sampleHasSize(v, size)) setSize('all');
  };
  const pickSize = (v: string) => {
    setSize(v);
    if (v !== 'all' && sample !== 'all' && !sizeHasSample(v, sample)) setSample('all');
  };

  const filtered = useMemo(() => {
    const stepLo = minStep === '' ? -Infinity : parseInt(minStep, 10);
    const stepHi = maxStep === '' ? Infinity : parseInt(maxStep, 10);
    let tLo = minT;
    let tHi = maxT;
    if (band !== 'all') {
      const start = parseInt(band.slice(1), 10) / 100;
      tLo = Math.max(tLo, start);
      tHi = Math.min(tHi, start + 0.1);
    }
    const out = previews.filter(p => {
      if (Number.isFinite(stepLo) && p.step < stepLo) return false;
      if (Number.isFinite(stepHi) && p.step > stepHi) return false;
      if (p.t < tLo || p.t > tHi) return false;
      // A specific sample selection only keeps previews with a matching
      // srcName, which means videos (no srcName) drop out unless "all".
      if (sample !== 'all' && p.srcName !== sample) return false;
      // Same shape for size: pre-suffix previews lack size and drop out
      // unless "all" — selecting a size means "show only this resolution".
      if (size !== 'all' && sizeKey(p.size) !== size) return false;
      return true;
    });
    const dir = sortDir === 'asc' ? 1 : -1;
    const keyFn = (p: DepthPreview): number => {
      if (sortKey === 'step') return p.step;
      if (sortKey === 't') return p.t;
      // `dc` is image-only; videos sort to the end ascending
      return p.dc ?? Number.POSITIVE_INFINITY;
    };
    out.sort((a, b) => (keyFn(a) - keyFn(b)) * dir);
    return out;
  }, [previews, minStep, maxStep, minT, maxT, band, sample, size, sortKey, sortDir]);

  const counts = useMemo(() => {
    const total = previews.length;
    const visible = filtered.length;
    return { total, visible };
  }, [previews, filtered]);

  // Index of the currently-selected preview within the *filtered* list. -1
  // when nothing is selected or when the selected item dropped out of the
  // visible set (e.g. filters changed). We use path-as-identity rather than
  // a raw index so filter/sort changes don't accidentally jump to a different
  // image.
  const selectedIdx = useMemo(() => {
    if (selectedPath == null) return -1;
    return filtered.findIndex(p => p.path === selectedPath);
  }, [selectedPath, filtered]);
  const selected = selectedIdx >= 0 ? filtered[selectedIdx] : null;

  // Keyboard nav for the overlay. Esc closes; left/right step through the
  // filtered list (wraps at the ends).
  useEffect(() => {
    if (selectedPath == null) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedPath(null);
      } else if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && filtered.length > 0) {
        const idx = selectedIdx >= 0 ? selectedIdx : 0;
        const delta = e.key === 'ArrowRight' ? 1 : -1;
        const next = (idx + delta + filtered.length) % filtered.length;
        setSelectedPath(filtered[next].path);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedPath, selectedIdx, filtered]);

  const emptyMessage = useMemo(() => {
    if (status === 'loading' && previews.length === 0) {
      return { icon: <LuLoader className="animate-spin w-8 h-8" />, title: 'Loading Depth Previews', sub: 'Please wait…' };
    }
    if (status === 'error') {
      return { icon: <LuBan className="w-8 h-8" />, title: 'Error Loading Depth Previews', sub: 'There was a problem fetching the previews.' };
    }
    if (status === 'success' && previews.length === 0) {
      return {
        icon: <LuImageOff className="w-8 h-8" />,
        title: 'No Depth Previews Yet',
        sub: 'Depth previews are written to <save_root>/depth_previews/ during training; check back after the first preview_every step.',
      };
    }
    if (previews.length > 0 && filtered.length === 0) {
      return { icon: <LuImageOff className="w-8 h-8" />, title: 'No matches', sub: 'No previews match the current filters.' };
    }
    return null;
  }, [status, previews.length, filtered.length]);

  const inputCls = 'bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200';
  const labelCls = 'text-xs text-gray-400 uppercase tracking-wide';

  return (
    <div ref={containerRef} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      {/* Filter / sort bar */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur border-b border-gray-800 px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className={labelCls}>Step</span>
          <input
            type="number"
            value={minStep}
            placeholder={stepBounds.lo ? `${stepBounds.lo}` : 'min'}
            onChange={e => setMinStep(e.target.value)}
            className={`${inputCls} w-20`}
          />
          <span className="text-gray-500">–</span>
          <input
            type="number"
            value={maxStep}
            placeholder={stepBounds.hi ? `${stepBounds.hi}` : 'max'}
            onChange={e => setMaxStep(e.target.value)}
            className={`${inputCls} w-20`}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className={labelCls}>T</span>
          <input
            type="number"
            step="0.05"
            min={0}
            max={1}
            value={minT}
            onChange={e => setMinT(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
            className={`${inputCls} w-16`}
            disabled={band !== 'all'}
          />
          <span className="text-gray-500">–</span>
          <input
            type="number"
            step="0.05"
            min={0}
            max={1}
            value={maxT}
            onChange={e => setMaxT(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
            className={`${inputCls} w-16`}
            disabled={band !== 'all'}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className={labelCls}>Band</span>
          <select value={band} onChange={e => setBand(e.target.value as Band)} className={inputCls}>
            {BAND_VALUES.map(b => (
              <option key={b} value={b}>
                {b === 'all' ? 'all' : `${b} (${(parseInt(b.slice(1)) / 100).toFixed(2)}–${(parseInt(b.slice(1)) / 100 + 0.1).toFixed(2)})`}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className={labelCls}>Sample</span>
          <select
            value={sample}
            onChange={e => pickSample(e.target.value)}
            className={`${inputCls} max-w-[14rem]`}
            disabled={sampleOptions.length === 0}
            title={sampleOptions.length === 0 ? 'No samples match the current Size filter' : 'Filter by source image'}
          >
            <option value="all">all ({sampleOptions.length})</option>
            {sampleOptions.map(s => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className={labelCls}>Size</span>
          <select
            value={size}
            onChange={e => pickSize(e.target.value)}
            className={inputCls}
            disabled={sizeOptions.length === 0}
            title={sizeOptions.length === 0 ? 'No sized previews match the current Sample filter (older trainer builds also omit the size suffix).' : 'Filter by sample resolution (W×H)'}
          >
            <option value="all">all ({sizeOptions.length})</option>
            {sizeOptions.map(s => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className={labelCls}>Sort</span>
          <select value={sortKey} onChange={e => setSortKey(e.target.value as SortKey)} className={inputCls}>
            <option value="step">step</option>
            <option value="t">t</option>
            <option value="dc">dc loss</option>
          </select>
          <button
            type="button"
            onClick={() => setSortDir(sortDir === 'asc' ? 'desc' : 'asc')}
            className={`${inputCls} cursor-pointer hover:bg-gray-800`}
            title="Toggle direction"
          >
            {sortDir === 'asc' ? '↑ asc' : '↓ desc'}
          </button>
        </div>
        <div className="flex-1" />
        <span className="text-xs text-gray-500">
          {counts.visible} / {counts.total} previews
        </span>
      </div>

      {/* Empty / loading / error state */}
      {emptyMessage && (
        <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-50 dark:bg-gray-800/50 text-gray-900 dark:text-gray-100 mx-auto max-w-md text-center">
          <div className="text-gray-500 dark:text-gray-400 mb-4">{emptyMessage.icon}</div>
          <h3 className="text-lg font-semibold mb-2">{emptyMessage.title}</h3>
          <p className="text-sm opacity-75 leading-relaxed">{emptyMessage.sub}</p>
        </div>
      )}

      {/* Grid */}
      {filtered.length > 0 && (
        <div className="p-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 gap-2">
          {filtered.map(p => (
            <div key={p.path} className="flex flex-col">
              <SampleImageCard
                imageUrl={p.path}
                numSamples={1}
                sampleImages={[p.path]}
                alt={`depth preview step ${p.step} t ${p.t}`}
                observerRoot={containerRef.current}
                onClick={() => setSelectedPath(p.path)}
              />
              <div className="bg-gray-900 text-xs text-gray-300 px-2 py-1 rounded-b-lg flex flex-wrap gap-x-3">
                <span><span className="text-gray-500">step</span> {p.step}</span>
                <span><span className="text-gray-500">t</span> {p.t.toFixed(2)}</span>
                {typeof p.dc === 'number' && (
                  <span><span className="text-gray-500">dc</span> {p.dc.toFixed(4)}</span>
                )}
                <span className="text-gray-500">{bandFor(p.t)}</span>
                {p.size && <span className="text-gray-500">{p.size.w}×{p.size.h}</span>}
                {p.srcName && <span className="text-gray-500 truncate">· {p.srcName}</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Zoom overlay */}
      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/85 flex flex-col items-center justify-center p-4"
          onClick={() => setSelectedPath(null)}
        >
          <button
            type="button"
            onClick={e => {
              e.stopPropagation();
              setSelectedPath(null);
            }}
            className="absolute top-3 right-3 w-10 h-10 rounded-full bg-gray-800/80 hover:bg-gray-700 text-gray-200 flex items-center justify-center"
            title="Close (Esc)"
            aria-label="Close"
          >
            <LuX className="w-5 h-5" />
          </button>
          <img
            src={`/api/img/${encodeURIComponent(selected.path)}`}
            alt={`depth preview step ${selected.step} t ${selected.t}`}
            className="max-w-[95vw] max-h-[85vh] object-contain rounded-md shadow-2xl"
            onClick={e => e.stopPropagation()}
          />
          <div
            className="mt-3 px-4 py-2 rounded-md bg-gray-900/90 text-gray-200 text-sm flex flex-wrap items-center gap-x-4"
            onClick={e => e.stopPropagation()}
          >
            <span><span className="text-gray-500">step</span> {selected.step}</span>
            <span><span className="text-gray-500">t</span> {selected.t.toFixed(2)}</span>
            {typeof selected.dc === 'number' && (
              <span><span className="text-gray-500">dc</span> {selected.dc.toFixed(4)}</span>
            )}
            <span><span className="text-gray-500">band</span> {bandFor(selected.t)}</span>
            {selected.size && (
              <span><span className="text-gray-500">size</span> {selected.size.w}×{selected.size.h}</span>
            )}
            {selected.srcName && (
              <span className="truncate max-w-[40ch]"><span className="text-gray-500">sample</span> {selected.srcName}</span>
            )}
            <span className="ml-auto text-gray-500 text-xs">
              {selectedIdx + 1} / {filtered.length}  ·  ← → to navigate, Esc to close
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
