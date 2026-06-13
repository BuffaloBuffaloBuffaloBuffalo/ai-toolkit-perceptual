import { NextResponse } from 'next/server';
import fs from 'fs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Ideogram structured captions are JSON. They can live next to the image under
// either a .txt or .json sidecar (the trainer reads them via the dataset's
// caption_ext). This route reads/writes that sidecar as raw text so the editor
// owns the JSON string as the single source of truth.

const CAPTION_EXTS = ['.txt', '.json'];

function captionPathFor(imgPath: string, ext: string): string {
  return imgPath.replace(/\.[^/.]+$/, '') + ext;
}

function normExt(raw: unknown): string {
  if (typeof raw !== 'string' || !raw.trim()) return '.txt';
  const e = raw.trim();
  const dotted = e.startsWith('.') ? e : '.' + e;
  return CAPTION_EXTS.includes(dotted) ? dotted : '.txt';
}

// GET-equivalent (POST so the body carries the absolute path like the fork's
// other img routes): returns { caption, ext } for the first existing sidecar,
// preferring the requested ext.
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const imgPath: string | undefined = body?.imgPath;
    const action: string = body?.action ?? 'read';

    if (!imgPath || typeof imgPath !== 'string' || imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    if (action === 'write') {
      const ext = normExt(body?.ext);
      const caption = typeof body?.caption === 'string' ? body.caption : '';
      fs.writeFileSync(captionPathFor(imgPath, ext), caption, 'utf-8');
      return NextResponse.json({ success: true, ext });
    }

    // read: prefer requested ext, then the other known exts.
    const preferred = normExt(body?.ext);
    const order = [preferred, ...CAPTION_EXTS.filter(e => e !== preferred)];
    for (const ext of order) {
      const p = captionPathFor(imgPath, ext);
      if (fs.existsSync(p)) {
        return NextResponse.json({ caption: fs.readFileSync(p, 'utf-8'), ext });
      }
    }
    // no sidecar yet
    return NextResponse.json({ caption: '', ext: preferred });
  } catch (e) {
    console.error('ideogram caption route error:', e);
    return NextResponse.json({ error: 'Caption operation failed' }, { status: 500 });
  }
}
