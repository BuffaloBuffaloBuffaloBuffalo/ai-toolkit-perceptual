import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

export interface DepthPreview {
  path: string;
  kind: 'image' | 'video';
  step: number;
  t: number;
  dc?: number;
  srcName?: string;
}

// Trainer writes to <save_root>/depth_previews/ with these formats:
//   image: `{src_name}_step{step:06d}_t{t:.2f}_dc{dc:.4f}.jpg`
//   video: `step{step:06d}_t{t:.2f}.webp`
// See extensions_built_in/sd_trainer/SDTrainer.py around the depth_previews dir.
const IMAGE_RE = /^(.+)_step(\d+)_t(\d+(?:\.\d+)?)_dc(\d+(?:\.\d+)?)\.jpg$/;
const VIDEO_RE = /^step(\d+)_t(\d+(?:\.\d+)?)\.webp$/;

function parseFilename(name: string): Omit<DepthPreview, 'path'> | null {
  const im = name.match(IMAGE_RE);
  if (im) {
    return {
      kind: 'image',
      srcName: im[1],
      step: parseInt(im[2], 10),
      t: parseFloat(im[3]),
      dc: parseFloat(im[4]),
    };
  }
  const vid = name.match(VIDEO_RE);
  if (vid) {
    return {
      kind: 'video',
      step: parseInt(vid[1], 10),
      t: parseFloat(vid[2]),
    };
  }
  return null;
}

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await (params as any);

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const previewsFolder = path.join(trainingFolder, job.name, 'depth_previews');
  if (!fs.existsSync(previewsFolder)) {
    return NextResponse.json({ previews: [] });
  }

  const previews: DepthPreview[] = fs
    .readdirSync(previewsFolder)
    .map(file => {
      const meta = parseFilename(file);
      if (!meta) return null;
      return { ...meta, path: path.join(previewsFolder, file) };
    })
    .filter((p): p is DepthPreview => p !== null);

  return NextResponse.json({ previews });
}
