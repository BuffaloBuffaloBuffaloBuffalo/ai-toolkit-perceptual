import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const RUN_ID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

export async function GET(
  _request: NextRequest,
  { params }: { params: { runId: string } },
) {
  try {
    const { runId } = await params;
    if (!RUN_ID_RE.test(runId)) {
      return NextResponse.json({ error: 'Invalid runId' }, { status: 400 });
    }

    const trainingRoot = await getTrainingFolder();
    const runDir = path.join(trainingRoot, 'dataset_face_detect', runId);

    if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
      return NextResponse.json({ error: 'Run not found' }, { status: 404 });
    }

    let progress: any = null;
    const progressPath = path.join(runDir, 'progress.json');
    if (fs.existsSync(progressPath)) {
      try {
        progress = JSON.parse(fs.readFileSync(progressPath, 'utf-8'));
      } catch {
        progress = { status: 'unknown', message: 'progress.json unparseable' };
      }
    }

    let cfg: any = null;
    const cfgPath = path.join(runDir, 'config.json');
    if (fs.existsSync(cfgPath)) {
      try {
        cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf-8'));
      } catch {
        cfg = null;
      }
    }

    const tiles: { name: string; path: string }[] = [];
    for (const entry of fs.readdirSync(runDir)) {
      if (entry.toLowerCase().endsWith('.png')) {
        tiles.push({ name: entry, path: path.join(runDir, entry) });
      }
    }
    tiles.sort((a, b) => a.name.localeCompare(b.name));

    const errors: string[] = fs
      .readdirSync(runDir)
      .filter(e => e.endsWith('.error.txt'))
      .sort();

    return NextResponse.json({
      runId,
      runDir,
      progress,
      config: cfg,
      tiles,
      errors,
      done: fs.existsSync(path.join(runDir, 'done.marker')),
    });
  } catch (err: any) {
    console.error('face-detect status error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
