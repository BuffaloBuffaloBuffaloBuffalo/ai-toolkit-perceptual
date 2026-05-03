import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const isWindows = process.platform === 'win32';

function resolvePython(): string {
  const venvCandidates = [
    isWindows
      ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'),
    isWindows
      ? path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python'),
  ];
  for (const cand of venvCandidates) {
    if (fs.existsSync(cand)) return cand;
  }
  return 'python';
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const datasetName: string | undefined = body.datasetName;
    const cfg = body.config ?? {};

    if (!datasetName || typeof datasetName !== 'string' || datasetName.includes('..') || datasetName.includes('/')) {
      return NextResponse.json({ error: 'Invalid datasetName' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const datasetDir = path.join(datasetsRoot, datasetName);
    if (!fs.existsSync(datasetDir) || !fs.statSync(datasetDir).isDirectory()) {
      return NextResponse.json({ error: `Dataset not found: ${datasetName}` }, { status: 404 });
    }

    const trainingRoot = await getTrainingFolder();
    const runId = uuidv4();
    const runDir = path.join(trainingRoot, 'dataset_face_detect', runId);
    fs.mkdirSync(runDir, { recursive: true });

    fs.writeFileSync(
      path.join(runDir, 'progress.json'),
      JSON.stringify({
        status: 'queued',
        message: 'Spawning Python worker...',
        done: 0, total: 0,
        dataset: datasetName,
      }),
    );

    const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'preflight_face_detection.py');
    if (!fs.existsSync(scriptPath)) {
      return NextResponse.json({ error: 'preflight_face_detection.py not found' }, { status: 500 });
    }

    const pythonPath = resolvePython();
    const args = [
      scriptPath,
      '--dataset-dir', datasetDir,
      '--output-dir', runDir,
      '--face-model', String(cfg.face_model ?? 'buffalo_l'),
      '--det-size', String(cfg.det_size ?? 640),
      '--limit', String(cfg.limit ?? 0),
    ];

    const subprocess = spawn(pythonPath, args, {
      cwd: TOOLKIT_ROOT,
      detached: true,
      stdio: 'ignore',
      env: { ...process.env },
      ...(isWindows ? { windowsHide: true } : {}),
    });
    if (subprocess.unref) subprocess.unref();

    try {
      fs.writeFileSync(path.join(runDir, 'pid.txt'), String(subprocess.pid ?? ''));
    } catch {
      // non-fatal
    }

    return NextResponse.json({ runId, runDir, datasetName });
  } catch (err: any) {
    console.error('face-detect start error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
