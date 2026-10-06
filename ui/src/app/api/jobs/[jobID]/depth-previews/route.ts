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
  /** Pixel dimensions of the underlying sample, when encoded in the filename
   *  (newer trainer builds only; older previews leave this undefined). */
  size?: { w: number; h: number };
}

// Trainer writes to <save_root>/depth_previews/ with these formats:
//   image: `{src_name}_step{step:06d}_t{t:.2f}_dc{dc:.4f}_s{W}x{H}.jpg`
//   video: `step{step:06d}_t{t:.2f}_s{W}x{H}.webp`
// The `_s{W}x{H}` segment is optional so previews from older builds that
// pre-date the size suffix still parse and surface in the UI.
// See extensions_built_in/sd_trainer/SDTrainer.py around the depth_previews dir.
const IMAGE_RE = /^(.+)_step(\d+)_t(\d+(?:\.\d+)?)_dc(\d+(?:\.\d+)?)(?:_s(\d+)x(\d+))?\.jpg$/;
const VIDEO_RE = /^step(\d+)_t(\d+(?:\.\d+)?)(?:_s(\d+)x(\d+))?\.webp$/;

function parseFilename(name: string): Omit<DepthPreview, 'path'> | null {
  const im = name.match(IMAGE_RE);
  if (im) {
    const meta: Omit<DepthPreview, 'path'> = {
      kind: 'image',
      srcName: im[1],
      step: parseInt(im[2], 10),
      t: parseFloat(im[3]),
      dc: parseFloat(im[4]),
    };
    if (im[5] && im[6]) meta.size = { w: parseInt(im[5], 10), h: parseInt(im[6], 10) };
    return meta;
  }
  const vid = name.match(VIDEO_RE);
  if (vid) {
    const meta: Omit<DepthPreview, 'path'> = {
      kind: 'video',
      step: parseInt(vid[1], 10),
      t: parseFloat(vid[2]),
    };
    if (vid[3] && vid[4]) meta.size = { w: parseInt(vid[3], 10), h: parseInt(vid[4], 10) };
    return meta;
  }
  return null;
}

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
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
