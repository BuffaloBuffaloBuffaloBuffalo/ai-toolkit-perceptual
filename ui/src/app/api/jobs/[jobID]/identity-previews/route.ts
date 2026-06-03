import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

export interface IdentityPreview {
  path: string;
  step: number;
  t: number;
  /** ArcFace cosine similarity between the generated x0 face and the reference.
   *  Can be negative, so the filename (and this field) may carry a leading `-`. */
  cos: number;
  srcName: string;
}

// Trainer writes to <save_root>/id_previews/ as:
//   `{src_name}_step{step:06d}_t{t:.2f}_cos{cos:.3f}.jpg`
// Each image is a combined [noisy | x0 | ArcFace crop] panel. Unlike depth
// previews there is no size suffix and no video variant — always a .jpg.
// cos is a cosine similarity in [-1, 1], so allow an optional leading minus.
// See extensions_built_in/sd_trainer/SDTrainer.py around the id_previews dir.
const IMAGE_RE = /^(.+)_step(\d+)_t(\d+(?:\.\d+)?)_cos(-?\d+(?:\.\d+)?)\.jpg$/;

function parseFilename(name: string): Omit<IdentityPreview, 'path'> | null {
  const im = name.match(IMAGE_RE);
  if (!im) return null;
  return {
    srcName: im[1],
    step: parseInt(im[2], 10),
    t: parseFloat(im[3]),
    cos: parseFloat(im[4]),
  };
}

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await (params as any);

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const previewsFolder = path.join(trainingFolder, job.name, 'id_previews');
  if (!fs.existsSync(previewsFolder)) {
    return NextResponse.json({ previews: [] });
  }

  const previews: IdentityPreview[] = fs
    .readdirSync(previewsFolder)
    .map(file => {
      const meta = parseFilename(file);
      if (!meta) return null;
      return { ...meta, path: path.join(previewsFolder, file) };
    })
    .filter((p): p is IdentityPreview => p !== null);

  return NextResponse.json({ previews });
}
