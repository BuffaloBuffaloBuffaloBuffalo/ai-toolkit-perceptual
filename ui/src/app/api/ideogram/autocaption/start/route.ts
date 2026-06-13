import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const isWindows = process.platform === 'win32';
const IMG_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'];

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

function countImages(dir: string): number {
  let n = 0;
  const walk = (d: string) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      if (e.name.startsWith('.')) continue;
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (IMG_EXTS.includes(path.extname(e.name).toLowerCase())) n++;
    }
  };
  walk(dir);
  return n;
}

// Spawn the Ideogram 4 auto-captioner (Qwen3-VL) over a dataset. Reuses the real
// captioner extension via run.py so captions match what training reads. Progress
// is tracked by counting caption sidecars (see the [runId] status route), so no
// job-DB hookup is needed.
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const datasetName: string | undefined = body?.datasetName;
    const captionExt: string = body?.captionExt === 'json' ? 'json' : 'txt';
    const recaption: boolean = body?.recaption === true;
    const model: string =
      typeof body?.model === 'string' && body.model.trim()
        ? body.model.trim()
        : 'Qwen/Qwen3-VL-8B-Instruct';

    if (!datasetName || typeof datasetName !== 'string' || datasetName.includes('..') || datasetName.includes('/')) {
      return NextResponse.json({ error: 'Invalid datasetName' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const datasetDir = path.join(datasetsRoot, datasetName);
    if (!fs.existsSync(datasetDir) || !fs.statSync(datasetDir).isDirectory()) {
      return NextResponse.json({ error: `Dataset not found: ${datasetName}` }, { status: 404 });
    }

    const total = countImages(datasetDir);
    if (total === 0) {
      return NextResponse.json({ error: 'No images in dataset' }, { status: 400 });
    }

    const trainingRoot = await getTrainingFolder();
    const runId = uuidv4();
    const runDir = path.join(trainingRoot, 'ideogram_caption', runId);
    fs.mkdirSync(runDir, { recursive: true });

    // Captioner job config consumed by run.py (matches the Ideogram4Captioner
    // extension's CaptionConfig fields).
    const captionerConfig = {
      job: 'extension',
      config: {
        name: `ideogram_caption_${runId}`,
        process: [
          {
            type: 'Ideogram4Captioner',
            model_name_or_path: model,
            path_to_caption: datasetDir,
            caption_extension: captionExt,
            recaption,
            device: 'cuda',
            dtype: 'bf16',
            quantize: false,
            max_res: 1024,
          },
        ],
      },
    };
    const configPath = path.join(runDir, 'captioner.json');
    fs.writeFileSync(configPath, JSON.stringify(captionerConfig, null, 2));

    const logStream = fs.openSync(path.join(runDir, 'caption.log'), 'a');
    const subprocess = spawn(resolvePython(), ['-u', path.join(TOOLKIT_ROOT, 'run.py'), configPath], {
      cwd: TOOLKIT_ROOT,
      detached: true,
      stdio: ['ignore', logStream, logStream],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      ...(isWindows ? { windowsHide: true } : {}),
    });
    if (subprocess.unref) subprocess.unref();

    fs.writeFileSync(
      path.join(runDir, 'meta.json'),
      JSON.stringify({
        pid: subprocess.pid ?? null,
        folder: datasetDir,
        captionExt,
        total,
        recaption,
        startedAt: Date.now(),
      }),
    );

    return NextResponse.json({ runId, total });
  } catch (err: any) {
    console.error('ideogram autocaption start error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
