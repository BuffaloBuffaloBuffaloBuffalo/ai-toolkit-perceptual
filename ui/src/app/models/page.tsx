'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaCloudUploadAlt, FaLink } from 'react-icons/fa';
import { LuLoader, LuX } from 'react-icons/lu';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { openConfirm } from '@/components/ConfirmModal';
import { apiClient } from '@/utils/api';

interface ModelEntry {
  filename: string;
  bytes: number;
  modifiedMs: number;
  path: string;
}

type TransferStatus = 'pending' | 'downloading' | 'done' | 'error' | 'cancelled';

interface Transfer {
  id: string;
  url: string;
  filename: string;
  status: TransferStatus;
  bytes: number;
  total: number | null;
  startedAt: number;
  finishedAt: number | null;
  error: string | null;
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0)} ${units[i]}`;
}
function formatDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleString();
}
function formatPercent(loaded: number, total: number | null): string {
  if (!total || total <= 0) return '—';
  return `${((loaded / total) * 100).toFixed(1)}%`;
}

const inputCls = 'w-full text-sm px-3 py-2 bg-gray-900 border border-gray-700 rounded-md text-gray-200 focus:ring-1 focus:ring-gray-500 focus:outline-none';
const labelCls = 'block text-xs text-gray-400 uppercase tracking-wide mb-1';
const buttonCls = 'inline-flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-sm font-medium text-white transition-colors';
const cardCls = 'bg-gray-900/60 border border-gray-800 rounded-lg p-4 flex flex-col';

export default function ModelsPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [root, setRoot] = useState<string>('');
  const [modelsStatus, setModelsStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [transfers, setTransfers] = useState<Transfer[]>([]);

  const refreshModels = useCallback(() => {
    setModelsStatus('loading');
    apiClient
      .get('/api/models')
      .then(res => {
        setModels(res.data.models ?? []);
        setRoot(res.data.root ?? '');
        setModelsStatus('success');
      })
      .catch(() => setModelsStatus('error'));
  }, []);

  const refreshTransfers = useCallback(() => {
    apiClient
      .get('/api/models/transfers')
      .then(res => setTransfers(res.data.transfers ?? []))
      .catch(() => {
        /* swallow */
      });
  }, []);

  useEffect(() => {
    refreshModels();
    refreshTransfers();
    const t = setInterval(refreshTransfers, 1500);
    return () => clearInterval(t);
  }, [refreshModels, refreshTransfers]);

  // Once a transfer completes, the model list should pick it up. We refresh
  // when any transfer flips from in-progress to a terminal state.
  const prevStatusRef = useRef<Record<string, TransferStatus>>({});
  useEffect(() => {
    let anyJustFinished = false;
    for (const t of transfers) {
      const prev = prevStatusRef.current[t.id];
      if (prev && prev !== t.status && (t.status === 'done' || t.status === 'cancelled' || t.status === 'error')) {
        anyJustFinished = true;
      }
      prevStatusRef.current[t.id] = t.status;
    }
    if (anyJustFinished) refreshModels();
  }, [transfers, refreshModels]);

  const totalBytes = useMemo(() => models.reduce((s, m) => s + m.bytes, 0), [models]);

  const handleDelete = (filename: string) => {
    openConfirm({
      title: 'Delete Model',
      message: `Delete "${filename}"? This permanently removes the file from disk.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .delete(`/api/models/${encodeURIComponent(filename)}`)
          .then(() => refreshModels())
          .catch(err => {
            // eslint-disable-next-line no-console
            console.error('Delete failed:', err);
          });
      },
    });
  };

  const columns: TableColumn[] = [
    {
      title: 'Filename',
      key: 'filename',
      render: row => <span className="text-gray-100 font-mono text-xs">{row.filename}</span>,
    },
    {
      title: 'Size',
      key: 'bytes',
      className: 'w-28 text-right tabular-nums',
      render: row => <span className="text-gray-300">{formatBytes(row.bytes)}</span>,
    },
    {
      title: 'Modified',
      key: 'modifiedMs',
      className: 'w-48',
      render: row => <span className="text-gray-400 text-xs">{formatDate(row.modifiedMs)}</span>,
    },
    {
      title: '',
      key: 'actions',
      className: 'w-12 text-right',
      render: row => (
        <button
          className="text-gray-300 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDelete(row.filename)}
          title="Delete"
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Models</h1>
        </div>
        <div className="flex-1" />
        <div className="text-xs text-gray-400 mr-3">
          {models.length} {models.length === 1 ? 'model' : 'models'} · {formatBytes(totalBytes)}
          {root && <span className="ml-3 text-gray-500 font-mono">{root}</span>}
        </div>
      </TopBar>
      <MainContent className="pt-20 px-6 pb-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <UploadCard onUploaded={refreshModels} />
          <UrlCard onQueued={refreshTransfers} />
        </div>

        {transfers.length > 0 && (
          <TransfersSection
            transfers={transfers}
            onChange={() => {
              refreshTransfers();
              refreshModels();
            }}
          />
        )}

        <div>
          <UniversalTable
            columns={columns}
            rows={models as unknown as Record<string, unknown>[]}
            isLoading={modelsStatus === 'loading'}
            onRefresh={refreshModels}
          />
          {modelsStatus === 'success' && models.length === 0 && (
            <div className="mt-4 text-center text-sm text-gray-500 py-6 border border-dashed border-gray-800 rounded-lg">
              No models yet. Upload one above, or paste a direct .safetensors URL.
            </div>
          )}
        </div>
      </MainContent>
    </>
  );
}

// ---------- Upload from disk ----------

function UploadCard({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // The RunPod/Cloudflare edge and the Next server both handle large bodies fine
  // (a 128 MiB POST clears server-side in ~1.5s); the real bottleneck is that ONE
  // TCP stream to the datacenter is bandwidth-delay-product limited — measured
  // ~13 Mbps while the uplink can do several× that. So we upload many fixed-size
  // chunks CONCURRENTLY to saturate the link: an `init` pre-allocates the file,
  // PARALLELISM chunks POST to their byte offsets at once, then we `finalize`. The
  // server records which offsets landed, so a reload/drop only re-sends the
  // missing chunks instead of restarting a multi-GB transfer.
  const CHUNK = 16 * 1024 * 1024; // 16 MiB per chunk — clears the edge, amortizes per-request overhead
  const PARALLELISM = 8; // concurrent chunk uploads — saturates a high-latency uplink one stream can't
  const MAX_CHUNK_RETRIES = 6; // per-chunk transient-failure retries before giving up

  const upload = async () => {
    if (!file || uploading) return;
    const f = file;
    setError(null);
    setStatus(null);
    setProgress(0);
    setUploading(true);

    // Mirror what apiClient adds — the project's auth interceptor lives in axios,
    // not in fetch, so we plumb the bearer token through ourselves on every call.
    const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
    const authH: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
    const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
    const post = (headers: Record<string, string>, body?: BodyInit | null) =>
      fetch('/api/models/upload', { method: 'POST', headers: { ...authH, ...headers }, body: body ?? undefined });
    // A "hard" error aborts the whole upload; everything else is retried.
    const hard = (msg: string) => Object.assign(new Error(msg), { hard: true });

    // Plan the chunks up front. A 0-byte file still gets one empty chunk so init +
    // finalize still produce the (empty) file.
    const chunks: { offset: number; size: number }[] = [];
    for (let o = 0; o < f.size; o += CHUNK) chunks.push({ offset: o, size: Math.min(CHUNK, f.size - o) });
    if (chunks.length === 0) chunks.push({ offset: 0, size: 0 });
    const totalBytes = f.size || 1;

    // Send one chunk, retrying transient failures (network, 5xx, 429, bodiless edge
    // bounce) with backoff. A vanished .part (NEED_INIT) re-inits then retries; a
    // hard 4xx (already exists, unsupported) aborts immediately.
    const sendOne = async (c: { offset: number; size: number }) => {
      for (let attempt = 0; ; attempt++) {
        try {
          const r = await post(
            {
              'X-Filename': f.name,
              'X-Chunk-Offset': String(c.offset),
              'X-File-Size': String(f.size),
              'Content-Type': 'application/octet-stream',
            },
            f.slice(c.offset, c.offset + c.size),
          );
          if (r.ok) return;
          const b = await r.json().catch(() => null);
          if (r.status === 409 && b?.code === 'NEED_INIT') {
            await post({ 'X-Filename': f.name, 'X-Upload-Init': '1', 'X-File-Size': String(f.size) });
          } else if (r.status === 409 && /exists/i.test(b?.error || '')) {
            throw hard('A model with that filename already exists');
          } else if (r.status >= 500 || r.status === 429 || !b) {
            if (attempt >= MAX_CHUNK_RETRIES) throw new Error(b?.error || `Chunk failed (HTTP ${r.status || 0})`);
          } else {
            throw hard(b?.error || `Chunk failed (HTTP ${r.status})`);
          }
        } catch (e: any) {
          if (e?.hard) throw e;
          if (attempt >= MAX_CHUNK_RETRIES) throw e;
        }
        await sleep(Math.min(8000, 500 * 2 ** Math.min(attempt, 4)));
      }
    };

    try {
      // Resume probe: bail if already present, else learn which offsets landed.
      const probeRes = await fetch(`/api/models/upload?filename=${encodeURIComponent(f.name)}`, { headers: authH });
      const probe = await probeRes.json().catch(() => ({} as any));
      if (probe?.exists) throw new Error('A model with that filename already exists');
      const received = new Set<number>(Array.isArray(probe?.received) ? probe.received.map(Number) : []);

      // Init a fresh upload (pre-allocate the .part). Skip if resuming into one.
      if (!probe?.partExists) {
        const r = await post({ 'X-Filename': f.name, 'X-Upload-Init': '1', 'X-File-Size': String(f.size) });
        if (!r.ok) {
          const b = await r.json().catch(() => null);
          throw new Error(b?.error || `Could not start upload (HTTP ${r.status})`);
        }
        received.clear();
      }

      let doneBytes = chunks.filter(c => received.has(c.offset)).reduce((s, c) => s + c.size, 0);
      setProgress(Math.min(1, doneBytes / totalBytes));
      const pending = chunks.filter(c => !received.has(c.offset));
      if (received.size > 0 && pending.length > 0) {
        setStatus(`Resuming — ${received.size}/${chunks.length} chunks already uploaded`);
      } else {
        setStatus(`Uploading ${chunks.length} chunk${chunks.length === 1 ? '' : 's'}, ${PARALLELISM} at a time…`);
      }

      // Worker pool: each worker pulls the next pending chunk until the queue drains.
      // `pending[idx++]` is atomic between awaits (JS is single-threaded), so no two
      // workers grab the same chunk. The first failure aborts the rest.
      let idx = 0;
      let aborted = false;
      const worker = async () => {
        for (;;) {
          if (aborted) return;
          const c = pending[idx++];
          if (!c) return;
          try {
            await sendOne(c);
          } catch (e) {
            aborted = true;
            throw e;
          }
          doneBytes += c.size;
          setProgress(Math.min(1, doneBytes / totalBytes));
        }
      };
      await Promise.all(Array.from({ length: Math.min(PARALLELISM, Math.max(1, pending.length)) }, worker));

      // Finalize: the server checks every expected offset landed and the .part is
      // the right size before renaming. If it reports gaps, re-send just those.
      for (let attempt = 0; ; attempt++) {
        const r = await post({
          'X-Filename': f.name,
          'X-Upload-Finalize': '1',
          'X-File-Size': String(f.size),
          'X-Chunk-Size': String(CHUNK),
        });
        if (r.ok) break;
        const b = await r.json().catch(() => null);
        if (r.status === 409 && b?.code === 'INCOMPLETE' && Array.isArray(b?.missing) && attempt < 3) {
          const miss = (b.missing as number[]).map(off => ({ offset: off, size: Math.min(CHUNK, f.size - off) }));
          setStatus(`Re-sending ${miss.length} missing chunk${miss.length === 1 ? '' : 's'}…`);
          for (const c of miss) await sendOne(c);
          continue;
        }
        throw new Error(b?.error || `Could not finalize upload (HTTP ${r.status})`);
      }

      setFile(null);
      setProgress(0);
      setStatus(null);
      if (inputRef.current) inputRef.current.value = '';
      setUploading(false);
      onUploaded();
    } catch (e: any) {
      setUploading(false);
      setStatus(null);
      setError(e?.message || 'Upload failed');
    }
  };

  return (
    <div className={cardCls}>
      <div className="flex items-center gap-2 mb-3">
        <FaCloudUploadAlt className="text-gray-300" />
        <h2 className="text-sm font-medium text-gray-200">Upload from disk</h2>
      </div>
      <p className="text-xs text-gray-500 mb-3">
        Pick a <code className="text-gray-400">.safetensors</code>, <code className="text-gray-400">.ckpt</code>,{' '}
        <code className="text-gray-400">.pt</code>, <code className="text-gray-400">.bin</code>, or{' '}
        <code className="text-gray-400">.gguf</code> file. Big multi-GB models are fine; they upload in
        chunks that auto-size to the connection and resume if it drops.
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".safetensors,.ckpt,.pt,.pth,.bin,.gguf"
        onChange={e => {
          setError(null);
          setFile(e.target.files?.[0] ?? null);
        }}
        className="text-xs text-gray-400 mb-3 file:mr-3 file:px-3 file:py-1 file:rounded file:border-0 file:bg-gray-800 file:text-gray-200 file:cursor-pointer"
      />
      {file && !uploading && (
        <div className="text-xs text-gray-400 mb-3">
          {file.name} · {formatBytes(file.size)}
        </div>
      )}
      {uploading && (
        <div className="mb-3">
          <div className="h-2 bg-gray-800 rounded overflow-hidden">
            <div className="h-full bg-blue-500 transition-all" style={{ width: `${(progress * 100).toFixed(1)}%` }} />
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {(progress * 100).toFixed(1)}% · {formatBytes(progress * (file?.size ?? 0))} of {formatBytes(file?.size ?? 0)}
          </div>
          {status && <div className="text-xs text-gray-500 mt-1">{status}</div>}
        </div>
      )}
      {error && <div className="text-xs text-red-400 mb-3">{error}</div>}
      <div className="flex-1" />
      <div className="flex justify-end">
        <Button className={buttonCls} disabled={!file || uploading} onClick={upload}>
          {uploading ? <LuLoader className="animate-spin" /> : <FaCloudUploadAlt />}
          {uploading ? 'Uploading' : 'Upload'}
        </Button>
      </div>
    </div>
  );
}

// ---------- Add from URL ----------

function UrlCard({ onQueued }: { onQueued: () => void }) {
  const [url, setUrl] = useState('');
  const [filename, setFilename] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = () => {
    if (!url || pending) return;
    setError(null);
    setPending(true);
    apiClient
      .post('/api/models/url', { url, filename: filename || undefined })
      .then(() => {
        setUrl('');
        setFilename('');
        onQueued();
      })
      .catch(err => {
        setError(err?.response?.data?.error ?? err?.message ?? 'Failed to queue download');
      })
      .finally(() => setPending(false));
  };

  return (
    <div className={cardCls}>
      <div className="flex items-center gap-2 mb-3">
        <FaLink className="text-gray-300" />
        <h2 className="text-sm font-medium text-gray-200">Add from URL</h2>
      </div>
      <p className="text-xs text-gray-500 mb-3">
        Paste a direct .safetensors / .ckpt URL (Civitai raw download, raw HuggingFace file URL, etc.). The server fetches it
        in the background; progress shows below.
      </p>
      <label className={labelCls}>URL</label>
      <input
        type="url"
        value={url}
        onChange={e => setUrl(e.target.value)}
        placeholder="https://…/model.safetensors"
        className={`${inputCls} mb-3`}
        disabled={pending}
      />
      <label className={labelCls}>Filename (optional)</label>
      <input
        type="text"
        value={filename}
        onChange={e => setFilename(e.target.value)}
        placeholder="auto-derive from URL"
        className={`${inputCls} mb-3`}
        disabled={pending}
      />
      {error && <div className="text-xs text-red-400 mb-3">{error}</div>}
      <div className="flex-1" />
      <div className="flex justify-end">
        <Button className={buttonCls} disabled={!url || pending} onClick={submit}>
          {pending ? <LuLoader className="animate-spin" /> : <FaLink />}
          {pending ? 'Queueing' : 'Start download'}
        </Button>
      </div>
    </div>
  );
}

// ---------- Active transfers ----------

function TransfersSection({ transfers, onChange }: { transfers: Transfer[]; onChange: () => void }) {
  const cancel = (id: string) => {
    apiClient.post('/api/models/transfers', { action: 'cancel', id }).finally(onChange);
  };
  const clearFinished = () => {
    apiClient.post('/api/models/transfers', { action: 'clearFinished' }).finally(onChange);
  };
  const anyFinished = transfers.some(t => t.status === 'done' || t.status === 'error' || t.status === 'cancelled');

  return (
    <div className={cardCls}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-200">Active transfers</h2>
        {anyFinished && (
          <button onClick={clearFinished} className="text-xs text-gray-400 hover:text-gray-200">
            Clear finished
          </button>
        )}
      </div>
      <div className="space-y-2">
        {transfers.map(t => {
          const inFlight = t.status === 'pending' || t.status === 'downloading';
          const pct = t.total ? Math.min(100, (t.bytes / t.total) * 100) : null;
          return (
            <div key={t.id} className="bg-gray-950/60 border border-gray-800 rounded-md px-3 py-2">
              <div className="flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-gray-100 font-mono truncate">{t.filename}</div>
                  <div className="text-xs text-gray-500 truncate">{t.url}</div>
                </div>
                <div className="text-xs text-gray-400 whitespace-nowrap">
                  <span className={
                    t.status === 'done' ? 'text-green-400' :
                    t.status === 'error' ? 'text-red-400' :
                    t.status === 'cancelled' ? 'text-yellow-500' :
                    'text-blue-400'
                  }>{t.status}</span>
                  {' · '}
                  {formatBytes(t.bytes)}
                  {t.total != null && ` / ${formatBytes(t.total)}`}
                  {pct != null && ` · ${formatPercent(t.bytes, t.total)}`}
                </div>
                {inFlight && (
                  <button
                    className="text-gray-400 hover:text-red-400 p-1"
                    onClick={() => cancel(t.id)}
                    title="Cancel"
                  >
                    <LuX className="w-4 h-4" />
                  </button>
                )}
              </div>
              {pct != null && (
                <div className="mt-2 h-1.5 bg-gray-800 rounded overflow-hidden">
                  <div
                    className={`h-full transition-all ${t.status === 'done' ? 'bg-green-500' : t.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              )}
              {t.error && <div className="mt-1 text-xs text-red-400">{t.error}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
