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

  // The RunPod edge (Cloudflare) rejects a request body over some size with a 400
  // before it reaches the app — a whole multi-GB file, and even a 50 MiB chunk,
  // both fail. We don't hardcode the cap (RunPod can change it): start optimistic
  // and, whenever the edge bounces a chunk, halve the chunk size and retry the
  // same bytes until it goes through, then keep that size. Each chunk POSTs to its
  // byte offset; the server appends, reports the authoritative offset (so we can
  // resume after a hiccup or reload), and renames into place once the .part
  // reaches the full file size.
  const START_CHUNK = 8 * 1024 * 1024; // 8 MiB — observed to clear the edge
  const MIN_CHUNK = 512 * 1024; // floor; below this we stop shrinking and back off
  const MAX_RETRIES = 8; // consecutive transient failures before giving up

  const upload = async () => {
    if (!file || uploading) return;
    const f = file;
    setError(null);
    setStatus(null);
    setProgress(0);
    setUploading(true);

    // Mirror what apiClient adds — the project's auth interceptor lives in axios,
    // not in raw XHR, so we plumb the bearer token through ourselves on the POSTs.
    const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
    const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

    type ChunkResult = { uploaded: number; complete: boolean };
    type ChunkError = Error & { kind: 'edge' | 'transient' | 'app'; status: number };
    const fail = (kind: ChunkError['kind'], message: string, httpStatus = 0): ChunkError =>
      Object.assign(new Error(message), { kind, status: httpStatus });

    const sendChunk = (offset: number, size: number) =>
      new Promise<ChunkResult>((resolve, reject) => {
        const end = Math.min(f.size, offset + size);
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', e => {
          if (e.lengthComputable) setProgress(Math.min(1, f.size ? (offset + e.loaded) / f.size : 1));
        });
        xhr.addEventListener('load', () => {
          const s = xhr.status;
          let body: any = null;
          try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
          if (s >= 200 && s < 300) {
            resolve({ uploaded: typeof body?.uploaded === 'number' ? body.uploaded : end, complete: !!body?.complete });
            return;
          }
          if (body && typeof body.error === 'string') {
            // The app answered — trust its verdict. Resync/transient codes retry;
            // hard 4xx (already exists, unsupported, unauthorized) abort.
            if (body.code === 'OFFSET_GAP' || s === 429 || s >= 500) { reject(fail('transient', body.error, s)); return; }
            reject(fail('app', body.error, s));
            return;
          }
          // No JSON (Cloudflare's HTML 400, or an opaque status) — the edge bounced
          // the body before the app saw it. Shrinking the chunk is the fix.
          reject(fail('edge', `Edge rejected a ${formatBytes(end - offset)} chunk (HTTP ${s || 0})`, s));
        });
        xhr.addEventListener('error', () => reject(fail('edge', 'Network/edge error', 0)));
        xhr.addEventListener('timeout', () => reject(fail('transient', 'Chunk timed out', 0)));
        xhr.open('POST', '/api/models/upload');
        if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        xhr.setRequestHeader('X-Filename', f.name);
        xhr.setRequestHeader('X-Chunk-Offset', String(offset));
        xhr.setRequestHeader('X-File-Size', String(f.size));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');
        xhr.send(f.slice(offset, end));
      });

    // Ask the server how much already landed, so a reload or retry resumes instead
    // of restarting. Throws 'app' if the model is already fully present.
    const serverOffset = async (): Promise<number> => {
      const res = await apiClient.get('/api/models/upload', { params: { filename: f.name } });
      if (res.data?.exists) throw fail('app', 'A model with that filename already exists', 409);
      const up = Number(res.data?.uploaded);
      return Number.isFinite(up) && up > 0 && up <= f.size ? up : 0;
    };

    try {
      let offset = await serverOffset();
      if (offset > 0) setStatus(`Resuming at ${formatBytes(offset)} of ${formatBytes(f.size)}`);
      setProgress(f.size ? Math.min(1, offset / f.size) : 0);
      let chunk = START_CHUNK;
      let fails = 0;

      // Loop until the server confirms completion (renamed into place). Sending a
      // zero-length chunk once offset has reached the end finalizes a full-but-
      // unrenamed .part or a 0-byte file.
      for (;;) {
        const before = offset;
        try {
          const res = await sendChunk(offset, offset < f.size ? chunk : 0);
          offset = res.uploaded;
          setProgress(Math.min(1, f.size ? offset / f.size : 1));
          if (res.complete) break;
          if (offset <= before) {
            // No forward progress on a success — avoid a hot loop.
            if (++fails > MAX_RETRIES) throw new Error('Upload stalled (no progress)');
            await sleep(Math.min(8000, 500 * 2 ** (fails - 1)));
          } else {
            fails = 0;
            setStatus(null);
          }
        } catch (err) {
          const e = err as ChunkError;
          if (e.kind === 'app') throw e; // unrecoverable — surface the message
          if (e.kind === 'edge' && chunk > MIN_CHUNK) {
            // Too big for the edge: halve and retry the same bytes. This is the
            // adaptive self-tune, not a real failure, so it spends no retry budget.
            chunk = Math.max(MIN_CHUNK, Math.floor(chunk / 2));
            setStatus(`Edge rejected a larger chunk; retrying at ${formatBytes(chunk)}…`);
            continue;
          }
          if (++fails > MAX_RETRIES) throw new Error(`${e.message} — gave up after ${MAX_RETRIES} retries`);
          // Transient (or edge already at the floor): back off, resync to the
          // server's offset, and retry so we never re-send bytes that landed.
          setStatus(`Hiccup (${e.message}); retry ${fails}/${MAX_RETRIES}…`);
          await sleep(Math.min(8000, 500 * 2 ** (fails - 1)));
          try { offset = await serverOffset(); setProgress(f.size ? Math.min(1, offset / f.size) : 0); } catch { /* keep offset */ }
        }
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
