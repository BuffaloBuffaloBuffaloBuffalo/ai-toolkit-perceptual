'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import { NumberInput, SelectInput, Checkbox } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { apiClient } from '@/utils/api';
import { Button } from '@headlessui/react';

type PreflightConfig = {
  segformer_res: number;
  body_close_radius: number;
  yolo_conf: number;
  primary_only: boolean;
  sam_size: 'tiny' | 'small' | 'base_plus' | 'large';
  limit: number;
};

type ProgressPayload = {
  status?: string;
  message?: string;
  total?: number;
  done?: number;
  current?: string;
  dataset?: string;
};

type Tile = { name: string; path: string };

type RunDetail = {
  runId: string;
  runDir: string;
  progress: ProgressPayload | null;
  config: any | null;
  tiles: Tile[];
  errors: string[];
  done: boolean;
};

type RunListItem = {
  runId: string;
  datasetName: string | null;
  status: string | null;
  total: number | null;
  done: number | null;
  mtime: number;
};

const DEFAULT_CFG: PreflightConfig = {
  segformer_res: 768,
  body_close_radius: 2,
  yolo_conf: 0.25,
  primary_only: true,
  sam_size: 'small',
  limit: 0,
};

export default function DatasetToolsPage() {
  const { datasets, status: dsStatus } = useDatasetList();
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [cfg, setCfg] = useState<PreflightConfig>(DEFAULT_CFG);
  const [runId, setRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Initial dataset selection once list loads.
  useEffect(() => {
    if (!selectedDataset && datasets.length > 0) {
      setSelectedDataset(datasets[0]);
    }
  }, [datasets, selectedDataset]);

  // Load prior runs once and after a successful start.
  const refreshRuns = async () => {
    try {
      const res = await apiClient.get('/api/dataset-tools/preflight');
      setRuns(res.data?.runs ?? []);
    } catch (e) {
      console.error('Failed to load runs', e);
    }
  };
  useEffect(() => {
    refreshRuns();
  }, []);

  // Poll active run.
  useEffect(() => {
    if (!runId) return;
    const tick = async () => {
      try {
        const res = await apiClient.get(`/api/dataset-tools/preflight/${runId}`);
        const detail = res.data as RunDetail;
        setRunDetail(detail);
        const st = detail.progress?.status;
        if (detail.done || st === 'done' || st === 'error') {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          refreshRuns();
        }
      } catch (e) {
        console.error('poll failed', e);
      }
    };
    tick(); // immediate first read
    pollRef.current = setInterval(tick, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [runId]);

  const handleRun = async () => {
    if (!selectedDataset) return;
    setErrorMsg(null);
    setSubmitting(true);
    setRunDetail(null);
    try {
      const res = await apiClient.post('/api/dataset-tools/preflight/start', {
        datasetName: selectedDataset,
        config: cfg,
      });
      setRunId(res.data.runId);
      refreshRuns();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.error ?? e?.message ?? 'Failed to start');
    } finally {
      setSubmitting(false);
    }
  };

  const datasetOptions = useMemo(
    () => datasets.map(d => ({ label: d, value: d })),
    [datasets],
  );
  const samOptions = [
    { label: 'tiny', value: 'tiny' },
    { label: 'small', value: 'small' },
    { label: 'base_plus', value: 'base_plus' },
    { label: 'large', value: 'large' },
  ];

  const progress = runDetail?.progress;
  const pct =
    progress?.total && progress.total > 0
      ? Math.round((100 * (progress.done ?? 0)) / progress.total)
      : 0;

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Dataset Tools</h1>
        </div>
      </TopBar>
      <MainContent>
        <div className="max-w-6xl space-y-6 pb-8">
          <section className="bg-gray-900 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-gray-100 mb-2">Subject Mask Preflight</h2>
            <p className="text-sm text-gray-400 mb-4">
              Run mask extraction on a dataset for visual QC. Tiles are written to{' '}
              <code className="text-gray-200">output/dataset_preflight/&lt;runId&gt;/</code>. Does
              not affect the per-image <code className="text-gray-200">_face_id_cache</code>.
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-2">
              <SelectInput
                label="Dataset"
                value={selectedDataset}
                onChange={setSelectedDataset}
                options={datasetOptions}
                disabled={dsStatus !== 'success'}
              />
              <NumberInput
                label="SegFormer Resolution"
                value={cfg.segformer_res}
                onChange={v => setCfg(c => ({ ...c, segformer_res: Number(v ?? DEFAULT_CFG.segformer_res) }))}
                min={256}
                max={2048}
              />
              <NumberInput
                label="Body Close Radius"
                value={cfg.body_close_radius}
                onChange={v => setCfg(c => ({ ...c, body_close_radius: Number(v ?? DEFAULT_CFG.body_close_radius) }))}
                min={0}
                max={12}
              />
              <NumberInput
                label="YOLO Confidence"
                value={cfg.yolo_conf}
                onChange={v => setCfg(c => ({ ...c, yolo_conf: Number(v ?? DEFAULT_CFG.yolo_conf) }))}
                min={0}
                max={1}
              />
              <SelectInput
                label="SAM Size (loaded but unused)"
                value={cfg.sam_size}
                onChange={v => setCfg(c => ({ ...c, sam_size: v as PreflightConfig['sam_size'] }))}
                options={samOptions}
              />
              <NumberInput
                label="Image Limit (0 = all)"
                value={cfg.limit}
                onChange={v => setCfg(c => ({ ...c, limit: Number(v ?? 0) }))}
                min={0}
              />
              <div className="col-span-2 md:col-span-3 pt-2">
                <Checkbox
                  label="Primary person only (largest YOLO box)"
                  checked={cfg.primary_only}
                  onChange={v => setCfg(c => ({ ...c, primary_only: v }))}
                />
              </div>
            </div>

            <div className="mt-4 flex items-center gap-3">
              <Button
                onClick={handleRun}
                disabled={!selectedDataset || submitting}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {submitting ? 'Starting…' : 'Run Preflight'}
              </Button>
              {runId && (
                <span className="text-xs text-gray-400">
                  runId: <code className="text-gray-200">{runId}</code>
                </span>
              )}
              {errorMsg && <span className="text-sm text-red-400">{errorMsg}</span>}
            </div>
          </section>

          {runDetail && (
            <section className="bg-gray-900 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-gray-100 mb-2">Active Run</h2>
              <div className="text-sm text-gray-300 mb-2">
                <span className="text-gray-400">status:</span> {progress?.status ?? '—'}{' '}
                <span className="text-gray-400 ml-3">message:</span> {progress?.message ?? '—'}
              </div>
              {progress?.total ? (
                <div className="w-full bg-gray-800 rounded-full h-2 mb-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${pct}%` }} />
                </div>
              ) : null}
              <div className="text-xs text-gray-500 mb-2">
                {progress?.done ?? 0} / {progress?.total ?? 0}
                {progress?.current ? ` — ${progress.current}` : ''}
                {runDetail.errors.length > 0 ? ` — ${runDetail.errors.length} error(s)` : ''}
              </div>

              {runDetail.tiles.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                  {runDetail.tiles.map(t => (
                    <figure key={t.name} className="bg-gray-950 rounded-md p-2">
                      <img
                        src={`/api/img/${encodeURIComponent(t.path)}`}
                        alt={t.name}
                        className="w-full h-auto rounded"
                        loading="lazy"
                      />
                      <figcaption className="text-xs text-gray-500 mt-1 truncate">{t.name}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </section>
          )}

          <section className="bg-gray-900 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold text-gray-100">Prior Runs</h2>
              <Button
                onClick={refreshRuns}
                className="text-xs text-gray-300 hover:text-white px-2 py-1 rounded"
              >
                Refresh
              </Button>
            </div>
            {runs.length === 0 ? (
              <div className="text-sm text-gray-500">No prior runs.</div>
            ) : (
              <div className="space-y-1">
                {runs.map(r => (
                  <button
                    key={r.runId}
                    onClick={() => setRunId(r.runId)}
                    className={`w-full text-left px-3 py-2 rounded hover:bg-gray-800 ${
                      r.runId === runId ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-200">
                        {r.datasetName ?? '(unknown dataset)'}{' '}
                        <span className="text-gray-500">— {r.status ?? 'unknown'}</span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {r.done ?? 0}/{r.total ?? 0}
                        {' · '}
                        {new Date(r.mtime).toLocaleString()}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 truncate">{r.runId}</div>
                  </button>
                ))}
              </div>
            )}
          </section>
        </div>
      </MainContent>
    </>
  );
}
