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
  mask_dilate_radius: number;
  skin_bias: number;
  yolo_conf: number;
  primary_only: boolean;
  sam_size: 'tiny' | 'small' | 'base_plus' | 'large';
  limit: number;
};

type FaceDetectConfig = {
  face_model: string;
  det_size: number;
  limit: number;
};

type DepthConfig = {
  depth_model: string;
  input_size: number;
  use_mask: boolean;
  segformer_res: number;
  body_close_radius: number;
  mask_dilate_radius: number;
  skin_bias: number;
  yolo_conf: number;
  primary_only: boolean;
  sam_size: 'tiny' | 'small' | 'base_plus' | 'large';
  dtype: 'fp16' | 'bf16' | 'fp32';
  limit: number;
};

type ProgressPayload = {
  status?: string;
  message?: string;
  total?: number;
  done?: number;
  current?: string;
  dataset?: string;
  detected?: number;
  failed?: number;
  padded?: number;
  processed?: number;
  empty_mask?: number;
  use_mask?: boolean;
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
  detected?: number | null;
  failed?: number | null;
  padded?: number | null;
  processed?: number | null;
  empty_mask?: number | null;
  use_mask?: boolean | null;
};

const DEFAULT_CFG: PreflightConfig = {
  segformer_res: 768,
  body_close_radius: 2,
  mask_dilate_radius: 0,
  skin_bias: 0,
  yolo_conf: 0.25,
  primary_only: true,
  sam_size: 'small',
  limit: 0,
};

const DEFAULT_FD_CFG: FaceDetectConfig = {
  face_model: 'buffalo_l',
  det_size: 640,
  limit: 0,
};

const DEFAULT_DEPTH_CFG: DepthConfig = {
  depth_model: 'depth-anything/Depth-Anything-V2-Small-hf',
  input_size: 518,
  use_mask: false,
  segformer_res: 768,
  body_close_radius: 2,
  mask_dilate_radius: 0,
  skin_bias: 0,
  yolo_conf: 0.25,
  primary_only: true,
  sam_size: 'small',
  dtype: 'fp16',
  limit: 0,
};

export default function DatasetToolsPage() {
  const { datasets, status: dsStatus } = useDatasetList();

  // ---------- Subject Mask Preflight state ----------
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [cfg, setCfg] = useState<PreflightConfig>(DEFAULT_CFG);
  const [runId, setRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---------- Face Detection Preflight state ----------
  const [fdDataset, setFdDataset] = useState<string>('');
  const [fdCfg, setFdCfg] = useState<FaceDetectConfig>(DEFAULT_FD_CFG);
  const [fdRunId, setFdRunId] = useState<string | null>(null);
  const [fdRunDetail, setFdRunDetail] = useState<RunDetail | null>(null);
  const [fdRuns, setFdRuns] = useState<RunListItem[]>([]);
  const [fdSubmitting, setFdSubmitting] = useState(false);
  const [fdErrorMsg, setFdErrorMsg] = useState<string | null>(null);
  const fdPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---------- Depth Preflight state ----------
  const [dpDataset, setDpDataset] = useState<string>('');
  const [dpCfg, setDpCfg] = useState<DepthConfig>(DEFAULT_DEPTH_CFG);
  const [dpRunId, setDpRunId] = useState<string | null>(null);
  const [dpRunDetail, setDpRunDetail] = useState<RunDetail | null>(null);
  const [dpRuns, setDpRuns] = useState<RunListItem[]>([]);
  const [dpSubmitting, setDpSubmitting] = useState(false);
  const [dpErrorMsg, setDpErrorMsg] = useState<string | null>(null);
  const dpPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Initial dataset selection once list loads.
  useEffect(() => {
    if (datasets.length > 0) {
      if (!selectedDataset) setSelectedDataset(datasets[0]);
      if (!fdDataset) setFdDataset(datasets[0]);
      if (!dpDataset) setDpDataset(datasets[0]);
    }
  }, [datasets, selectedDataset, fdDataset, dpDataset]);

  const refreshRuns = async () => {
    try {
      const res = await apiClient.get('/api/dataset-tools/preflight');
      setRuns(res.data?.runs ?? []);
    } catch (e) {
      console.error('Failed to load runs', e);
    }
  };
  const refreshFdRuns = async () => {
    try {
      const res = await apiClient.get('/api/dataset-tools/face-detect');
      setFdRuns(res.data?.runs ?? []);
    } catch (e) {
      console.error('Failed to load face-detect runs', e);
    }
  };
  const refreshDpRuns = async () => {
    try {
      const res = await apiClient.get('/api/dataset-tools/depth');
      setDpRuns(res.data?.runs ?? []);
    } catch (e) {
      console.error('Failed to load depth runs', e);
    }
  };
  useEffect(() => {
    refreshRuns();
    refreshFdRuns();
    refreshDpRuns();
  }, []);

  // Poll active subject-mask run.
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
    tick();
    pollRef.current = setInterval(tick, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [runId]);

  // Poll active face-detect run.
  useEffect(() => {
    if (!fdRunId) return;
    const tick = async () => {
      try {
        const res = await apiClient.get(`/api/dataset-tools/face-detect/${fdRunId}`);
        const detail = res.data as RunDetail;
        setFdRunDetail(detail);
        const st = detail.progress?.status;
        if (detail.done || st === 'done' || st === 'error') {
          if (fdPollRef.current) {
            clearInterval(fdPollRef.current);
            fdPollRef.current = null;
          }
          refreshFdRuns();
        }
      } catch (e) {
        console.error('face-detect poll failed', e);
      }
    };
    tick();
    fdPollRef.current = setInterval(tick, 2000);
    return () => {
      if (fdPollRef.current) clearInterval(fdPollRef.current);
      fdPollRef.current = null;
    };
  }, [fdRunId]);

  // Poll active depth run.
  useEffect(() => {
    if (!dpRunId) return;
    const tick = async () => {
      try {
        const res = await apiClient.get(`/api/dataset-tools/depth/${dpRunId}`);
        const detail = res.data as RunDetail;
        setDpRunDetail(detail);
        const st = detail.progress?.status;
        if (detail.done || st === 'done' || st === 'error') {
          if (dpPollRef.current) {
            clearInterval(dpPollRef.current);
            dpPollRef.current = null;
          }
          refreshDpRuns();
        }
      } catch (e) {
        console.error('depth poll failed', e);
      }
    };
    tick();
    dpPollRef.current = setInterval(tick, 2000);
    return () => {
      if (dpPollRef.current) clearInterval(dpPollRef.current);
      dpPollRef.current = null;
    };
  }, [dpRunId]);

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

  const handleFdRun = async () => {
    if (!fdDataset) return;
    setFdErrorMsg(null);
    setFdSubmitting(true);
    setFdRunDetail(null);
    try {
      const res = await apiClient.post('/api/dataset-tools/face-detect/start', {
        datasetName: fdDataset,
        config: fdCfg,
      });
      setFdRunId(res.data.runId);
      refreshFdRuns();
    } catch (e: any) {
      setFdErrorMsg(e?.response?.data?.error ?? e?.message ?? 'Failed to start');
    } finally {
      setFdSubmitting(false);
    }
  };

  const handleDpRun = async () => {
    if (!dpDataset) return;
    setDpErrorMsg(null);
    setDpSubmitting(true);
    setDpRunDetail(null);
    try {
      const res = await apiClient.post('/api/dataset-tools/depth/start', {
        datasetName: dpDataset,
        config: dpCfg,
      });
      setDpRunId(res.data.runId);
      refreshDpRuns();
    } catch (e: any) {
      setDpErrorMsg(e?.response?.data?.error ?? e?.message ?? 'Failed to start');
    } finally {
      setDpSubmitting(false);
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
  const faceModelOptions = [
    { label: 'buffalo_l (default)', value: 'buffalo_l' },
    { label: 'buffalo_m', value: 'buffalo_m' },
    { label: 'buffalo_s', value: 'buffalo_s' },
    { label: 'antelopev2', value: 'antelopev2' },
  ];
  const depthModelOptions = [
    { label: 'DA2 Small (default)', value: 'depth-anything/Depth-Anything-V2-Small-hf' },
    { label: 'DA2 Base', value: 'depth-anything/Depth-Anything-V2-Base-hf' },
    { label: 'DA2 Large', value: 'depth-anything/Depth-Anything-V2-Large-hf' },
  ];
  const dtypeOptions = [
    { label: 'fp16', value: 'fp16' },
    { label: 'bf16', value: 'bf16' },
    { label: 'fp32', value: 'fp32' },
  ];

  const progress = runDetail?.progress;
  const pct =
    progress?.total && progress.total > 0
      ? Math.round((100 * (progress.done ?? 0)) / progress.total)
      : 0;

  const fdProgress = fdRunDetail?.progress;
  const fdPct =
    fdProgress?.total && fdProgress.total > 0
      ? Math.round((100 * (fdProgress.done ?? 0)) / fdProgress.total)
      : 0;

  const dpProgress = dpRunDetail?.progress;
  const dpPct =
    dpProgress?.total && dpProgress.total > 0
      ? Math.round((100 * (dpProgress.done ?? 0)) / dpProgress.total)
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
                label="Body Close Radius (fills holes)"
                value={cfg.body_close_radius}
                onChange={v => setCfg(c => ({ ...c, body_close_radius: Number(v ?? DEFAULT_CFG.body_close_radius) }))}
                min={0}
                max={12}
              />
              <NumberInput
                label="Mask Dilate Radius (grows boundary)"
                value={cfg.mask_dilate_radius}
                onChange={v => setCfg(c => ({ ...c, mask_dilate_radius: Number(v ?? DEFAULT_CFG.mask_dilate_radius) }))}
                min={0}
                max={64}
              />
              <NumberInput
                label="Skin Bias (push skin → body)"
                value={cfg.skin_bias}
                onChange={v => setCfg(c => ({ ...c, skin_bias: Number(v ?? DEFAULT_CFG.skin_bias) }))}
                min={0}
                max={8}
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
              <h2 className="text-lg font-semibold text-gray-100 mb-2">Active Subject Mask Run</h2>
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
              <h2 className="text-lg font-semibold text-gray-100">Prior Subject Mask Runs</h2>
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

          {/* ====================== Face Detection Preflight ====================== */}

          <section className="bg-gray-900 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-gray-100 mb-2">Face Detection Preflight</h2>
            <p className="text-sm text-gray-400 mb-4">
              Run InsightFace detection (RetinaFace) on a dataset for visual QC. Each tile shows
              the original image with bbox + keypoints overlaid; orange means a tight close-up
              triggered the padding fallback. Tiles are written to{' '}
              <code className="text-gray-200">output/dataset_face_detect/&lt;runId&gt;/</code>.
              Does not write to <code className="text-gray-200">_face_id_cache</code>.
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-2">
              <SelectInput
                label="Dataset"
                value={fdDataset}
                onChange={setFdDataset}
                options={datasetOptions}
                disabled={dsStatus !== 'success'}
              />
              <SelectInput
                label="Face Model"
                value={fdCfg.face_model}
                onChange={v => setFdCfg(c => ({ ...c, face_model: v as string }))}
                options={faceModelOptions}
              />
              <NumberInput
                label="Detection Size"
                value={fdCfg.det_size}
                onChange={v => setFdCfg(c => ({ ...c, det_size: Number(v ?? DEFAULT_FD_CFG.det_size) }))}
                min={160}
                max={1920}
              />
              <NumberInput
                label="Image Limit (0 = all)"
                value={fdCfg.limit}
                onChange={v => setFdCfg(c => ({ ...c, limit: Number(v ?? 0) }))}
                min={0}
              />
            </div>

            <div className="mt-4 flex items-center gap-3">
              <Button
                onClick={handleFdRun}
                disabled={!fdDataset || fdSubmitting}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {fdSubmitting ? 'Starting…' : 'Run Face Detection'}
              </Button>
              {fdRunId && (
                <span className="text-xs text-gray-400">
                  runId: <code className="text-gray-200">{fdRunId}</code>
                </span>
              )}
              {fdErrorMsg && <span className="text-sm text-red-400">{fdErrorMsg}</span>}
            </div>
          </section>

          {fdRunDetail && (
            <section className="bg-gray-900 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-gray-100 mb-2">Active Face Detection Run</h2>
              <div className="text-sm text-gray-300 mb-2">
                <span className="text-gray-400">status:</span> {fdProgress?.status ?? '—'}{' '}
                <span className="text-gray-400 ml-3">message:</span> {fdProgress?.message ?? '—'}
              </div>
              {fdProgress?.total ? (
                <div className="w-full bg-gray-800 rounded-full h-2 mb-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${fdPct}%` }} />
                </div>
              ) : null}
              <div className="text-xs text-gray-500 mb-2">
                {fdProgress?.done ?? 0} / {fdProgress?.total ?? 0}
                {fdProgress?.current ? ` — ${fdProgress.current}` : ''}
                {fdRunDetail.errors.length > 0 ? ` — ${fdRunDetail.errors.length} error(s)` : ''}
              </div>
              {(fdProgress?.detected != null || fdProgress?.failed != null || fdProgress?.padded != null) && (
                <div className="text-xs text-gray-400 mb-2">
                  <span className="text-green-400">detected: {fdProgress?.detected ?? 0}</span>
                  {' · '}
                  <span className="text-orange-400">padded fallback: {fdProgress?.padded ?? 0}</span>
                  {' · '}
                  <span className="text-red-400">failed: {fdProgress?.failed ?? 0}</span>
                </div>
              )}

              {fdRunDetail.tiles.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                  {fdRunDetail.tiles.map(t => (
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
              <h2 className="text-lg font-semibold text-gray-100">Prior Face Detection Runs</h2>
              <Button
                onClick={refreshFdRuns}
                className="text-xs text-gray-300 hover:text-white px-2 py-1 rounded"
              >
                Refresh
              </Button>
            </div>
            {fdRuns.length === 0 ? (
              <div className="text-sm text-gray-500">No prior runs.</div>
            ) : (
              <div className="space-y-1">
                {fdRuns.map(r => (
                  <button
                    key={r.runId}
                    onClick={() => setFdRunId(r.runId)}
                    className={`w-full text-left px-3 py-2 rounded hover:bg-gray-800 ${
                      r.runId === fdRunId ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-200">
                        {r.datasetName ?? '(unknown dataset)'}{' '}
                        <span className="text-gray-500">— {r.status ?? 'unknown'}</span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {r.detected != null ? (
                          <>
                            <span className="text-green-400">{r.detected}</span>
                            {r.padded ? <span className="text-orange-400"> ({r.padded} padded)</span> : null}
                            {r.failed ? <span className="text-red-400"> · {r.failed} failed</span> : null}
                            {' · '}
                          </>
                        ) : null}
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

          {/* ====================== Depth Preflight ====================== */}

          <section className="bg-gray-900 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-gray-100 mb-2">
              Depth Preflight {dpCfg.use_mask ? '(Depth + Subject Mask)' : '(Depth only)'}
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              Run Depth-Anything-V2 on a dataset for visual QC. Without the mask toggle each
              tile is <code className="text-gray-200">[ original | depth ]</code>; with mask on
              it becomes <code className="text-gray-200">[ original | depth | subject mask | depth × mask ]</code>{' '}
              so you can verify the spatial region the depth-consistency loss is restricted to.
              Tiles are written to{' '}
              <code className="text-gray-200">output/dataset_depth/&lt;runId&gt;/</code>. Does
              not write to <code className="text-gray-200">_face_id_cache</code>.
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-2">
              <SelectInput
                label="Dataset"
                value={dpDataset}
                onChange={setDpDataset}
                options={datasetOptions}
                disabled={dsStatus !== 'success'}
              />
              <SelectInput
                label="Depth Model"
                value={dpCfg.depth_model}
                onChange={v => setDpCfg(c => ({ ...c, depth_model: v as string }))}
                options={depthModelOptions}
              />
              <NumberInput
                label="DA2 Input Size"
                value={dpCfg.input_size}
                onChange={v => setDpCfg(c => ({ ...c, input_size: Number(v ?? DEFAULT_DEPTH_CFG.input_size) }))}
                min={140}
                max={1400}
              />
              <SelectInput
                label="dtype"
                value={dpCfg.dtype}
                onChange={v => setDpCfg(c => ({ ...c, dtype: v as DepthConfig['dtype'] }))}
                options={dtypeOptions}
              />
              <NumberInput
                label="Image Limit (0 = all)"
                value={dpCfg.limit}
                onChange={v => setDpCfg(c => ({ ...c, limit: Number(v ?? 0) }))}
                min={0}
              />
              <div className="col-span-2 md:col-span-3 pt-2">
                <Checkbox
                  label="Apply subject mask (loads YOLO + SegFormer; slower)"
                  checked={dpCfg.use_mask}
                  onChange={v => setDpCfg(c => ({ ...c, use_mask: v }))}
                />
              </div>

              {dpCfg.use_mask && (
                <>
                  <NumberInput
                    label="SegFormer Resolution"
                    value={dpCfg.segformer_res}
                    onChange={v => setDpCfg(c => ({ ...c, segformer_res: Number(v ?? DEFAULT_DEPTH_CFG.segformer_res) }))}
                    min={256}
                    max={2048}
                  />
                  <NumberInput
                    label="Body Close Radius (fills holes)"
                    value={dpCfg.body_close_radius}
                    onChange={v => setDpCfg(c => ({ ...c, body_close_radius: Number(v ?? DEFAULT_DEPTH_CFG.body_close_radius) }))}
                    min={0}
                    max={12}
                  />
                  <NumberInput
                    label="Mask Dilate Radius (grows boundary)"
                    value={dpCfg.mask_dilate_radius}
                    onChange={v => setDpCfg(c => ({ ...c, mask_dilate_radius: Number(v ?? DEFAULT_DEPTH_CFG.mask_dilate_radius) }))}
                    min={0}
                    max={64}
                  />
                  <NumberInput
                    label="Skin Bias (push skin → body)"
                    value={dpCfg.skin_bias}
                    onChange={v => setDpCfg(c => ({ ...c, skin_bias: Number(v ?? DEFAULT_DEPTH_CFG.skin_bias) }))}
                    min={0}
                    max={8}
                  />
                  <NumberInput
                    label="YOLO Confidence"
                    value={dpCfg.yolo_conf}
                    onChange={v => setDpCfg(c => ({ ...c, yolo_conf: Number(v ?? DEFAULT_DEPTH_CFG.yolo_conf) }))}
                    min={0}
                    max={1}
                  />
                  <SelectInput
                    label="SAM Size (loaded but unused)"
                    value={dpCfg.sam_size}
                    onChange={v => setDpCfg(c => ({ ...c, sam_size: v as DepthConfig['sam_size'] }))}
                    options={samOptions}
                  />
                  <div className="col-span-2 md:col-span-3 pt-2">
                    <Checkbox
                      label="Primary person only (largest YOLO box)"
                      checked={dpCfg.primary_only}
                      onChange={v => setDpCfg(c => ({ ...c, primary_only: v }))}
                    />
                  </div>
                </>
              )}
            </div>

            <div className="mt-4 flex items-center gap-3">
              <Button
                onClick={handleDpRun}
                disabled={!dpDataset || dpSubmitting}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-md"
              >
                {dpSubmitting ? 'Starting…' : 'Run Depth Preflight'}
              </Button>
              {dpRunId && (
                <span className="text-xs text-gray-400">
                  runId: <code className="text-gray-200">{dpRunId}</code>
                </span>
              )}
              {dpErrorMsg && <span className="text-sm text-red-400">{dpErrorMsg}</span>}
            </div>
          </section>

          {dpRunDetail && (
            <section className="bg-gray-900 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-gray-100 mb-2">Active Depth Run</h2>
              <div className="text-sm text-gray-300 mb-2">
                <span className="text-gray-400">status:</span> {dpProgress?.status ?? '—'}{' '}
                <span className="text-gray-400 ml-3">message:</span> {dpProgress?.message ?? '—'}
              </div>
              {dpProgress?.total ? (
                <div className="w-full bg-gray-800 rounded-full h-2 mb-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${dpPct}%` }} />
                </div>
              ) : null}
              <div className="text-xs text-gray-500 mb-2">
                {dpProgress?.done ?? 0} / {dpProgress?.total ?? 0}
                {dpProgress?.current ? ` — ${dpProgress.current}` : ''}
                {dpRunDetail.errors.length > 0 ? ` — ${dpRunDetail.errors.length} error(s)` : ''}
              </div>
              {dpProgress?.use_mask && (dpProgress?.empty_mask ?? 0) > 0 && (
                <div className="text-xs text-orange-400 mb-2">
                  Empty subject mask on {dpProgress?.empty_mask} image(s)
                </div>
              )}

              {dpRunDetail.tiles.length > 0 && (
                <div className="grid grid-cols-1 gap-3 mt-3">
                  {dpRunDetail.tiles.map(t => (
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
              <h2 className="text-lg font-semibold text-gray-100">Prior Depth Runs</h2>
              <Button
                onClick={refreshDpRuns}
                className="text-xs text-gray-300 hover:text-white px-2 py-1 rounded"
              >
                Refresh
              </Button>
            </div>
            {dpRuns.length === 0 ? (
              <div className="text-sm text-gray-500">No prior runs.</div>
            ) : (
              <div className="space-y-1">
                {dpRuns.map(r => (
                  <button
                    key={r.runId}
                    onClick={() => setDpRunId(r.runId)}
                    className={`w-full text-left px-3 py-2 rounded hover:bg-gray-800 ${
                      r.runId === dpRunId ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-200">
                        {r.datasetName ?? '(unknown dataset)'}{' '}
                        <span className="text-gray-500">
                          — {r.status ?? 'unknown'}
                          {r.use_mask ? ' · with mask' : ' · depth only'}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {r.empty_mask ? <span className="text-orange-400">{r.empty_mask} empty mask · </span> : null}
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
