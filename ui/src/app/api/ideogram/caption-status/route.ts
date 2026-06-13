import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const IMG_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'];
const CAPTION_EXTS = ['.txt', '.json'];

// Returns the absolute paths of images in a dataset that have a non-empty
// caption sidecar (.txt or .json) — used to mark "captioned" images in the grid.
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetName: string | undefined = body?.datasetName;
    if (!datasetName || typeof datasetName !== 'string' || datasetName.includes('..') || datasetName.includes('/')) {
      return NextResponse.json({ error: 'Invalid datasetName' }, { status: 400 });
    }
    const datasetDir = path.join(await getDatasetsRoot(), datasetName);
    if (!fs.existsSync(datasetDir)) {
      return NextResponse.json({ captioned: [] });
    }

    const captioned: string[] = [];
    const walk = (d: string) => {
      for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        if (e.name.startsWith('.')) continue;
        const p = path.join(d, e.name);
        if (e.isDirectory()) {
          walk(p);
          continue;
        }
        if (!IMG_EXTS.includes(path.extname(e.name).toLowerCase())) continue;
        const base = p.replace(/\.[^/.]+$/, '');
        for (const ext of CAPTION_EXTS) {
          const sidecar = base + ext;
          try {
            if (fs.existsSync(sidecar) && fs.readFileSync(sidecar, 'utf-8').trim().length > 0) {
              captioned.push(p);
              break;
            }
          } catch {
            // ignore unreadable sidecar
          }
        }
      }
    };
    walk(datasetDir);
    return NextResponse.json({ captioned });
  } catch (err: any) {
    console.error('ideogram caption-status error:', err);
    return NextResponse.json({ error: err?.message || 'Internal error' }, { status: 500 });
  }
}
