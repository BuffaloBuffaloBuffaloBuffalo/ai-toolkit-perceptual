import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';

const RUN_ID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

export async function GET() {
  try {
    const trainingRoot = await getTrainingFolder();
    const root = path.join(trainingRoot, 'dataset_depth');
    if (!fs.existsSync(root)) {
      return NextResponse.json({ runs: [] });
    }

    const runs: Array<{
      runId: string;
      datasetName: string | null;
      status: string | null;
      total: number | null;
      done: number | null;
      processed: number | null;
      empty_mask: number | null;
      use_mask: boolean | null;
      mtime: number;
    }> = [];

    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory() || !RUN_ID_RE.test(entry.name)) continue;
      const runDir = path.join(root, entry.name);
      const progressPath = path.join(runDir, 'progress.json');
      let datasetName: string | null = null;
      let status: string | null = null;
      let total: number | null = null;
      let doneCt: number | null = null;
      let processed: number | null = null;
      let emptyMask: number | null = null;
      let useMask: boolean | null = null;
      try {
        if (fs.existsSync(progressPath)) {
          const p = JSON.parse(fs.readFileSync(progressPath, 'utf-8'));
          datasetName = p.dataset ?? null;
          status = p.status ?? null;
          total = p.total ?? null;
          doneCt = p.done ?? null;
          processed = p.processed ?? null;
          emptyMask = p.empty_mask ?? null;
          useMask = typeof p.use_mask === 'boolean' ? p.use_mask : null;
        }
      } catch {
        // skip unparseable
      }
      runs.push({
        runId: entry.name,
        datasetName,
        status,
        total,
        done: doneCt,
        processed,
        empty_mask: emptyMask,
        use_mask: useMask,
        mtime: fs.statSync(runDir).mtimeMs,
      });
    }

    runs.sort((a, b) => b.mtime - a.mtime);
    return NextResponse.json({ runs });
  } catch (err: any) {
    console.error('depth list error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
