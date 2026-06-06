import { NextRequest, NextResponse } from 'next/server';
import { createWriteStream, existsSync, unlinkSync, statSync } from 'node:fs';
import { mkdir, rename, open, appendFile, readFile, writeFile, unlink } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { getModelsRoot } from '@/server/settings';

const MODEL_EXTENSIONS = new Set(['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf']);

// Sanitize (don't reject) so real-world model names — spaces, parentheses,
// "Realistic Vision V5 (fp16).safetensors" — upload fine. path.basename strips
// any directory component (no traversal), disallowed chars collapse to '_', and
// we only hard-fail when there's no recognized model extension to anchor on.
function sanitizeFilename(raw: string): string | null {
  const base = path.basename(raw);
  if (!base || base === '.' || base === '..') return null;
  const ext = path.extname(base).toLowerCase();
  if (!MODEL_EXTENSIONS.has(ext)) return null;
  let stem = base
    .slice(0, base.length - ext.length)
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '');
  if (!stem) stem = 'model';
  return `${stem}${ext}`.slice(0, 255);
}

function resolvePaths(modelsDir: string, filename: string) {
  const target = path.join(modelsDir, filename);
  return { target, partPath: `${target}.part`, donePath: `${target}.part.done` };
}

// The ".done" sidecar is an append-only log of the byte offsets the server has
// fully written. Appends of a short "<offset>\n" record are atomic on POSIX
// (well under PIPE_BUF), so many concurrent chunk requests can record progress
// without locking or clobbering each other. We read it back as a de-duped set.
async function readDoneOffsets(donePath: string): Promise<number[]> {
  try {
    const txt = await readFile(donePath, 'utf8');
    const set = new Set<number>();
    for (const line of txt.split('\n')) {
      const n = parseInt(line, 10);
      if (Number.isFinite(n)) set.add(n);
    }
    return [...set];
  } catch {
    return [];
  }
}

// GET /api/models/upload?filename=foo.safetensors
// Resume probe: reports whether the model already exists, whether an in-progress
// .part is present, and which chunk offsets have already landed — so the client
// re-sends only the missing chunks after a disconnect or reload instead of
// restarting a multi-GB transfer from zero.
export async function GET(request: NextRequest) {
  const rawName = request.nextUrl.searchParams.get('filename');
  if (!rawName) {
    return NextResponse.json({ error: 'filename query param is required' }, { status: 400 });
  }
  const filename = sanitizeFilename(rawName);
  if (!filename) {
    return NextResponse.json({ error: `Unsupported model file "${rawName}".` }, { status: 400 });
  }
  const modelsDir = await getModelsRoot();
  const { target, partPath, donePath } = resolvePaths(modelsDir, filename);
  if (existsSync(target)) {
    return NextResponse.json({ filename, exists: true, complete: true, uploaded: 0, partExists: false, received: [] });
  }
  const partExists = existsSync(partPath);
  const received = partExists ? await readDoneOffsets(donePath) : [];
  // `uploaded` is kept for shape compatibility; the parallel client resumes from
  // `received` (a pre-allocated .part is full-size on disk from the start, so its
  // size is not a progress signal).
  return NextResponse.json({ filename, exists: false, complete: false, uploaded: 0, partExists, received });
}

// Streaming, CHUNKED, PARALLEL, RESUMABLE upload. The RunPod/Cloudflare edge and
// the Next server both handle large bodies fine; the bottleneck is that a single
// TCP stream to the datacenter is bandwidth-delay-product limited, so the client
// uploads many fixed-size chunks CONCURRENTLY to saturate the uplink. Because
// chunks now arrive out of order, the server can't assume contiguous growth:
//
//   1. init     (X-Upload-Init)     — create/truncate the .part to the full size
//                                      (sparse) and reset the .done sidecar.
//   2. chunk    (X-Chunk-Offset)    — write the body at its byte offset (an
//                                      in-bounds overwrite of the pre-allocated
//                                      file, so concurrent non-overlapping writes
//                                      are safe) and record the offset in .done.
//   3. finalize (X-Upload-Finalize) — verify every expected offset landed and the
//                                      .part is the right size, then rename it.
//
// Writing at a fixed offset makes a re-sent chunk idempotent, and we keep the
// .part/.done on error so a drop costs a retry of the missing chunks, not a
// multi-GB restart. A request with neither offset nor X-File-Size is treated as a
// single whole-file upload (back-compatible).
export async function POST(request: NextRequest) {
  const rawName = request.headers.get('x-filename');
  if (!rawName) {
    return NextResponse.json({ error: 'X-Filename header is required' }, { status: 400 });
  }
  const filename = sanitizeFilename(rawName);
  if (!filename) {
    return NextResponse.json(
      { error: `Unsupported model file "${rawName}". Use a .safetensors/.ckpt/.pt/.pth/.bin/.gguf file.` },
      { status: 400 },
    );
  }

  const modelsDir = await getModelsRoot();
  await mkdir(modelsDir, { recursive: true });
  const { target, partPath, donePath } = resolvePaths(modelsDir, filename);

  if (existsSync(target)) {
    return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
  }

  const fileSizeHeader = request.headers.get('x-file-size');
  const fileSize = fileSizeHeader != null ? Math.max(0, parseInt(fileSizeHeader, 10) || 0) : null;

  // ---- init: pre-allocate the .part to the full size and reset progress ----
  if (request.headers.get('x-upload-init')) {
    if (fileSize == null) {
      return NextResponse.json({ error: 'X-File-Size header is required for init' }, { status: 400 });
    }
    const fh = await open(partPath, 'w');
    try {
      await fh.truncate(fileSize);
    } finally {
      await fh.close();
    }
    await writeFile(donePath, '');
    return NextResponse.json({ ok: true, init: true, filename });
  }

  // ---- finalize: confirm completeness, then rename into place ----
  if (request.headers.get('x-upload-finalize')) {
    if (fileSize == null) {
      return NextResponse.json({ error: 'X-File-Size header is required for finalize' }, { status: 400 });
    }
    if (!existsSync(partPath)) {
      return NextResponse.json({ error: 'No in-progress upload to finalize', code: 'NEED_INIT' }, { status: 409 });
    }
    const chunkSize = Math.max(0, parseInt(request.headers.get('x-chunk-size') ?? '0', 10) || 0);
    const got = new Set(await readDoneOffsets(donePath));
    const missing: number[] = [];
    if (chunkSize > 0) {
      for (let o = 0; o < fileSize; o += chunkSize) {
        if (!got.has(o)) missing.push(o);
      }
    }
    if (missing.length) {
      return NextResponse.json(
        { error: `Upload incomplete: ${missing.length} chunk(s) missing`, code: 'INCOMPLETE', missing },
        { status: 409 },
      );
    }
    const size = statSync(partPath).size;
    if (size !== fileSize) {
      return NextResponse.json(
        { error: `Size mismatch: have ${size}, expected ${fileSize}`, code: 'SIZE_MISMATCH', uploaded: size },
        { status: 409 },
      );
    }
    if (existsSync(target)) {
      try { unlinkSync(partPath); } catch { /* ignore */ }
      try { unlinkSync(donePath); } catch { /* ignore */ }
      return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
    }
    await rename(partPath, target);
    try { await unlink(donePath); } catch { /* ignore */ }
    return NextResponse.json({ ok: true, complete: true, filename, uploaded: size, path: target });
  }

  if (!request.body) {
    return NextResponse.json({ error: 'Request body is empty' }, { status: 400 });
  }

  const offsetHeader = request.headers.get('x-chunk-offset');

  // ---- single-shot whole-file upload (back-compatible: no offset, no size) ----
  if (offsetHeader == null && fileSize == null) {
    try {
      const stream = Readable.fromWeb(request.body as any);
      const ws = createWriteStream(partPath, { flags: 'w' });
      await pipeline(stream, ws);
      const uploaded = statSync(partPath).size;
      if (existsSync(target)) {
        try { unlinkSync(partPath); } catch { /* ignore */ }
        return NextResponse.json({ error: 'A model with that filename already exists' }, { status: 409 });
      }
      await rename(partPath, target);
      return NextResponse.json({ ok: true, complete: true, filename, uploaded, path: target });
    } catch (e: any) {
      return NextResponse.json({ error: e?.message ?? 'Upload failed' }, { status: 500 });
    }
  }

  // ---- chunk: write the body at its byte offset into the pre-allocated .part ----
  const offset = Math.max(0, parseInt(offsetHeader ?? '0', 10) || 0);
  // A chunk needs the .part that init created. If it's gone (e.g. a stale resume),
  // tell the client to re-init rather than silently creating a wrong-sized file.
  if (!existsSync(partPath)) {
    return NextResponse.json({ error: 'Upload not initialized', code: 'NEED_INIT' }, { status: 409 });
  }
  try {
    const stream = Readable.fromWeb(request.body as any);
    const ws = createWriteStream(partPath, { flags: 'r+', start: offset });
    await pipeline(stream, ws);
    await appendFile(donePath, `${offset}\n`);
    return NextResponse.json({ ok: true, complete: false, filename, offset });
  } catch (e: any) {
    // Keep the .part/.done: the client re-sends only the missing chunks.
    return NextResponse.json({ error: e?.message ?? 'Upload failed' }, { status: 500 });
  }
}
