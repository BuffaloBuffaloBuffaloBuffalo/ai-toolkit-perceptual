import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';

const RUN_ID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

export async function GET() {
  try {
    const trainingRoot = await getTrainingFolder();
    const root = path.join(trainingRoot, 'dataset_face_detect');
    if (!fs.existsSync(root)) {
      return NextResponse.json({ runs: [] });
    }

    const runs: Array<{
      runId: string;
      datasetName: string | null;
      status: string | null;
      total: number | null;
      done: number | null;
      detected: number | null;
      failed: number | null;
      padded: number | null;
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
      let detected: number | null = null;
      let failed: number | null = null;
      let padded: number | null = null;
      try {
        if (fs.existsSync(progressPath)) {
          const p = JSON.parse(fs.readFileSync(progressPath, 'utf-8'));
          datasetName = p.dataset ?? null;
          status = p.status ?? null;
          total = p.total ?? null;
          doneCt = p.done ?? null;
          detected = p.detected ?? null;
          failed = p.failed ?? null;
          padded = p.padded ?? null;
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
        detected,
        failed,
        padded,
        mtime: fs.statSync(runDir).mtimeMs,
      });
    }

    runs.sort((a, b) => b.mtime - a.mtime);
    return NextResponse.json({ runs });
  } catch (err: any) {
    console.error('face-detect list error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
