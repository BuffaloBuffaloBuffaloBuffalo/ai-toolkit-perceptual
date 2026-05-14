'use client';

import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';

/**
 * Drop-in replacement for `useState` that mirrors its value into
 * `window.localStorage` under the given key. Reads on mount; writes on every
 * value change. Returns the same `[value, setValue]` tuple as `useState`, so
 * function-updaters (`setX(prev => …)`) still work.
 *
 * Failures (quota exceeded, storage disabled, JSON parse error) are silently
 * swallowed — the in-memory state still works, you just lose persistence.
 *
 * Keys should namespace the consumer (e.g. `"aitk:metrics:<jobId>:view"`) to
 * avoid collisions and to scope per-job state.
 */
export default function useLocalStorageState<T>(
  key: string,
  defaultValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return defaultValue;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw == null) return defaultValue;
      return JSON.parse(raw) as T;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // quota / disabled — fall through
    }
  }, [key, value]);

  return [value, setValue];
}
