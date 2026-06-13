import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const RUN_ID_RE = /^[0-9a-fA-F-]{8,}$/;
const IMG_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'];

function isAlive(pid: number | null): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0); // signal 0 = liveness probe
    return true;
  } catch {
    return false;
  }
}

// Count images that have a caption sidecar with the given extension.
function countCaptioned(dir: string, ext: string): number {
  const dotted = ext.startsWith('.') ? ext : '.' + ext;
  let n = 0;
  const walk = (d: string) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      if (e.name.startsWith('.')) continue;
      const p = path.join(d, e.name);
      if (e.isDirectory()) {
        walk(p);
      } else if (IMG_EXTS.includes(path.extname(e.name).toLowerCase())) {
        const sidecar = p.replace(/\.[^/.]+$/, '') + dotted;
        if (fs.existsSync(sidecar)) n++;
      }
    }
  };
  walk(dir);
  return n;
}

export async function GET(_request: NextRequest, { params }: { params: { runId: string } }) {
  try {
    const { runId } = await params;
    if (!RUN_ID_RE.test(runId)) {
      return NextResponse.json({ error: 'Invalid runId' }, { status: 400 });
    }
    const trainingRoot = await getTrainingFolder();
    const runDir = path.join(trainingRoot, 'ideogram_caption', runId);
    const metaPath = path.join(runDir, 'meta.json');
    if (!fs.existsSync(metaPath)) {
      return NextResponse.json({ error: 'Run not found' }, { status: 404 });
    }
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    const alive = isAlive(meta.pid);
    const captioned = fs.existsSync(meta.folder) ? countCaptioned(meta.folder, meta.captionExt) : 0;

    // Tail the captioner log for surfacing errors / status text.
    let tail = '';
    const logPath = path.join(runDir, 'caption.log');
    if (fs.existsSync(logPath)) {
      const buf = fs.readFileSync(logPath, 'utf-8');
      tail = buf.slice(-1500);
    }

    const done = !alive;
    const status = alive ? 'running' : captioned >= (meta.total ?? 0) ? 'completed' : 'stopped';
    return NextResponse.json({
      runId,
      status,
      done,
      captioned,
      total: meta.total ?? 0,
      captionExt: meta.captionExt,
      logTail: tail,
    });
  } catch (err: any) {
    console.error('ideogram autocaption status error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
