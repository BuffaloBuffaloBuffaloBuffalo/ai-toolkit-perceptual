import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  if (!fs.existsSync(logPath)) {
    return NextResponse.json({ keys: [], key: 'loss', points: [] });
  }

  const url = new URL(request.url);
  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  const db = openDb(logPath);

  try {
    const keysRows = await all<{ key: string }>(db, `SELECT key FROM metric_keys ORDER BY key ASC`);
    const keys = keysRows.map((r) => r.key);

    const points = await all<{
      step: number;
      wall_time: number;
      value: number | null;
      value_text: string | null;
    }>(
      db,
      `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.value_real AS value,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      WHERE m.key = ?
        AND (? IS NULL OR m.step > ?)
        AND (m.step % ?) = 0
      ORDER BY m.step ASC
      LIMIT ?
      `,
      [key, sinceStep, sinceStep, stride, limit]
    );

    return NextResponse.json({
      key,
      keys,
      points: points.map((p) => {
        // value_text now carries either a stringified scalar (legacy
        // fallback) or a JSON breakdown payload from the metrics-overhaul
        // MetricBuffer, of shape:
        //   {"samples": [...], "n": ..., "mean": ..., "std": ...}
        // We expose it as `breakdown` whenever it parses as an object so
        // the chart tooltip can render the per-sample drill-down.
        let breakdown: Record<string, unknown> | null = null;
        let fallbackValue: number | null = p.value;
        if (p.value_text != null) {
          const trimmed = p.value_text.trim();
          if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            try {
              const parsed = JSON.parse(trimmed);
              if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                breakdown = parsed as Record<string, unknown>;
              }
            } catch {
              breakdown = null;
            }
          } else if (fallbackValue == null) {
            const asNum = Number(trimmed);
            if (!Number.isNaN(asNum)) fallbackValue = asNum;
          }
        }
        return {
          step: p.step,
          wall_time: p.wall_time,
          value: fallbackValue,
          ...(breakdown != null ? { breakdown } : {}),
        };
      }),
    });
  } finally {
    await closeDb(db);
  }
}
