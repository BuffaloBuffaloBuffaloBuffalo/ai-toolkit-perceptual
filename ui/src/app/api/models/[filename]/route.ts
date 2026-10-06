import { NextRequest, NextResponse } from 'next/server';
import { existsSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { getModelsRoot } from '@/server/settings';

function sanitizeFilename(raw: string): string | null {
  const base = path.basename(raw);
  if (!base || base === '.' || base === '..') return null;
  if (base !== raw) return null;
  if (!/^[A-Za-z0-9._-]{1,255}$/.test(base)) return null;
  return base;
}

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ filename: string }> }) {
  const { filename: raw } = await (params as any);
  const filename = sanitizeFilename(decodeURIComponent(raw));
  if (!filename) {
    return NextResponse.json({ error: 'Invalid filename' }, { status: 400 });
  }
  const modelsDir = await getModelsRoot();
  const target = path.join(modelsDir, filename);
  // Defence-in-depth: refuse paths that resolve outside the models dir even
  // though sanitizeFilename should have blocked anything funky already.
  if (!target.startsWith(modelsDir + path.sep) && target !== modelsDir) {
    return NextResponse.json({ error: 'Path escape rejected' }, { status: 400 });
  }
  if (!existsSync(target)) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }
  try {
    unlinkSync(target);
    return NextResponse.json({ ok: true });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? 'Delete failed' }, { status: 500 });
  }
}
