'use client';

import { useMemo, useRef, useState } from 'react';
import useDepthPreviews, { DepthPreview } from '@/hooks/useDepthPreviews';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';

type SortKey = 'step' | 't' | 'dc';
type SortDir = 'asc' | 'desc';

// Bands match the trainer's convention: bin_start = floor(t * 10) / 10, label
// `t{int(bin_start*100):02d}`. See SDTrainer.py around line 2147 etc.
const BAND_VALUES = ['all', 't00', 't10', 't20', 't30', 't40', 't50', 't60', 't70', 't80', 't90'] as const;
type Band = (typeof BAND_VALUES)[number];

function bandFor(t: number): Band {
  const lo = Math.floor(Math.max(0, Math.min(0.999999, t)) * 10) * 10;
  return `t${lo.toString().padStart(2, '0')}` as Band;
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

  // Filter / sort state. Empty strings on the step inputs mean "no constraint"
  // — keeps the controls usable before previews finish loading.
  const [minStep, setMinStep] = useState<string>('');
  const [maxStep, setMaxStep] = useState<string>('');
  const [minT, setMinT] = useState<number>(0);
  const [maxT, setMaxT] = useState<number>(1);
  const [band, setBand] = useState<Band>('all');
  const [sortKey, setSortKey] = useState<SortKey>('step');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

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
  }, [previews, minStep, maxStep, minT, maxT, band, sortKey, sortDir]);

  const counts = useMemo(() => {
    const total = previews.length;
    const visible = filtered.length;
    return { total, visible };
  }, [previews, filtered]);

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
          <span className={labelCls}>Sort</span>
          <select value={sortKey} onChange={e => setSortKey(e.target.value as SortKey)} className={inputCls}>
            <option value="step">step</option>
            <option value="t">t</option>
            <option value="dc">dc loss</option>
          </select>
          <button
            type="button"
            onClick={() => setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))}
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
              />
              <div className="bg-gray-900 text-xs text-gray-300 px-2 py-1 rounded-b-lg flex flex-wrap gap-x-3">
                <span><span className="text-gray-500">step</span> {p.step}</span>
                <span><span className="text-gray-500">t</span> {p.t.toFixed(2)}</span>
                {typeof p.dc === 'number' && (
                  <span><span className="text-gray-500">dc</span> {p.dc.toFixed(4)}</span>
                )}
                <span className="text-gray-500">{bandFor(p.t)}</span>
                {p.srcName && <span className="text-gray-500 truncate">· {p.srcName}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
