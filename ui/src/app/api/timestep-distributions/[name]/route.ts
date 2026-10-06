import { NextRequest, NextResponse } from 'next/server';
import { deleteCurve, loadCurve } from '@/server/timestepCurves';

export async function GET(_request: NextRequest, { params }: { params: Promise<{ name: string }> }) {
  const { name } = await (params as any);
  const curve = await loadCurve('distribution', decodeURIComponent(name));
  if (!curve) return NextResponse.json({ error: 'Not found' }, { status: 404 });
  return NextResponse.json({ curve });
}

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ name: string }> }) {
  const { name } = await (params as any);
  const ok = await deleteCurve('distribution', decodeURIComponent(name));
  if (!ok) return NextResponse.json({ error: 'Not found or invalid name' }, { status: 404 });
  return NextResponse.json({ ok: true });
}
