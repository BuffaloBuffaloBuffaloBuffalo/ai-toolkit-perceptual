'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import classNames from 'classnames';
import { SquareDashed, X, Wand2, RefreshCw, ImageOff } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { BoundingBoxEditor, extractBoxes } from '@/components/BoundingBoxOverlay';
import IdeogramCaptionSidebar, { isIdeogramCaption } from '@/components/IdeogramCaptionSidebar';
import UpsamplePromptsModal, { openUpsamplePromptsModal } from '@/components/UpsamplePromptsModal';

interface ImgItem {
  img_path: string;
}

function safeParse(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// Empty Ideogram caption skeleton, so a plain/blank caption becomes editable.
function seedCaption(highLevel: string): string {
  return JSON.stringify(
    {
      high_level_description: highLevel.trim(),
      style_description: { aesthetics: '', lighting: '', photo: '', medium: '' },
      compositional_deconstruction: { background: '', elements: [] },
    },
    null,
    2,
  );
}

const imgUrl = (absPath: string) => `/api/img/${encodeURIComponent(absPath)}`;

export default function IdeogramPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>('');
  const [images, setImages] = useState<ImgItem[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [activeImg, setActiveImg] = useState<string | null>(null);

  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [captionExt, setCaptionExt] = useState<string>('.txt');
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);

  // Load dataset list once.
  useEffect(() => {
    apiClient
      .get('/api/datasets/list')
      .then(res => setDatasets(Array.isArray(res.data) ? res.data : []))
      .catch(err => console.error('Failed to list datasets', err));
  }, []);

  const loadImages = useCallback((ds: string) => {
    if (!ds) return;
    setLoadingImages(true);
    setActiveImg(null);
    apiClient
      .post('/api/datasets/listImages', { datasetName: ds })
      .then(res => {
        const imgs: ImgItem[] = res.data?.images ?? [];
        imgs.sort((a, b) => a.img_path.localeCompare(b.img_path));
        setImages(imgs);
      })
      .catch(err => {
        console.error('Failed to list images', err);
        setImages([]);
      })
      .finally(() => setLoadingImages(false));
  }, []);

  useEffect(() => {
    if (dataset) loadImages(dataset);
  }, [dataset, loadImages]);

  // Load the caption sidecar for the active image.
  const openImage = useCallback((absPath: string) => {
    setActiveImg(absPath);
    setSelectedBoxIndex(null);
    setIsDrawing(false);
    setShowBoxes(true);
    apiClient
      .post('/api/ideogram/caption', { imgPath: absPath, action: 'read' })
      .then(res => {
        const text: string = res.data?.caption ?? '';
        setCaption(text);
        setSavedCaption(text);
        if (res.data?.ext) setCaptionExt(res.data.ext);
      })
      .catch(err => {
        console.error('Failed to read caption', err);
        setCaption('');
        setSavedCaption('');
      });
  }, []);

  const isIdeogram = useMemo(() => isIdeogramCaption(caption), [caption]);
  const isDirty = caption.trim() !== savedCaption.trim();
  const editBoxes = useMemo(() => extractBoxes(safeParse(caption)), [caption]);

  const save = useCallback(() => {
    if (!activeImg) return;
    apiClient
      .post('/api/ideogram/caption', {
        imgPath: activeImg,
        action: 'write',
        ext: captionExt,
        caption,
      })
      .then(() => setSavedCaption(caption))
      .catch(err => console.error('Failed to save caption', err));
  }, [activeImg, caption, captionExt]);

  // Mutate the caption JSON's element array (local state only).
  const editCaption = useCallback(
    (fn: (elements: any[], data: any) => any): any => {
      const data = safeParse(caption);
      if (!data) return undefined;
      const elements = data?.compositional_deconstruction?.elements;
      if (!Array.isArray(elements)) return undefined;
      const result = fn(elements, data);
      setCaption(JSON.stringify(data, null, 2));
      return result;
    },
    [caption],
  );

  const handleBoxChange = useCallback(
    (elementIndex: number, box: { y1: number; x1: number; y2: number; x2: number }) => {
      editCaption(els => {
        if (els[elementIndex]) els[elementIndex].bbox = [box.y1, box.x1, box.y2, box.x2];
      });
    },
    [editCaption],
  );

  const handleCreateBox = useCallback(
    (box: { y1: number; x1: number; y2: number; x2: number }) => {
      const newIndex = editCaption(els => {
        els.push({ type: 'obj', bbox: [box.y1, box.x1, box.y2, box.x2], desc: '' });
        return els.length - 1;
      });
      setSelectedBoxIndex(typeof newIndex === 'number' ? newIndex : null);
      setIsDrawing(false);
    },
    [editCaption],
  );

  // Upsample a rough description into a structured caption via the modal.
  const upsample = useCallback(() => {
    const current = safeParse(caption);
    const seed = current?.high_level_description ?? (isIdeogram ? '' : caption);
    openUpsamplePromptsModal(
      [{ index: 0, prompt: seed || '', aspectRatio: 'auto' }],
      (_i, newPrompt) => {
        setCaption(newPrompt);
        setShowBoxes(true);
      },
    );
  }, [caption, isIdeogram]);

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-3 w-full">
          <h1 className="text-lg font-semibold text-gray-100">Ideogram Captions</h1>
          <select
            value={dataset}
            onChange={e => setDataset(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 outline-none focus:border-blue-500"
          >
            <option value="">Select a dataset…</option>
            {datasets.map(d => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          {dataset && (
            <button
              type="button"
              onClick={() => loadImages(dataset)}
              title="Reload images"
              className="text-gray-400 hover:text-gray-200"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}
          {activeImg && (
            <button
              type="button"
              onClick={() => setActiveImg(null)}
              className="ml-auto text-xs text-gray-400 hover:text-gray-200 border border-gray-700 rounded px-2 py-1"
            >
              ← Back to grid
            </button>
          )}
        </div>
      </TopBar>

      <MainContent>
        {!dataset && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2">
            <ImageOff className="w-8 h-8" />
            <span>Select a dataset to edit Ideogram captions.</span>
          </div>
        )}

        {/* Thumbnail grid */}
        {dataset && !activeImg && (
          <div className="p-3">
            {loadingImages ? (
              <div className="text-gray-500">Loading images…</div>
            ) : images.length === 0 ? (
              <div className="text-gray-500">No images found in this dataset.</div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
                {images.map(img => (
                  <button
                    key={img.img_path}
                    type="button"
                    onClick={() => openImage(img.img_path)}
                    className="group relative aspect-square overflow-hidden rounded-lg border border-gray-800 hover:border-blue-500 transition-colors"
                    title={img.img_path}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={imgUrl(img.img_path)}
                      alt=""
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Editor */}
        {dataset && activeImg && (
          <div className="flex flex-col lg:flex-row h-full overflow-hidden">
            <div className="relative flex-1 min-w-0 flex items-center justify-center bg-gray-900 overflow-hidden p-4">
              <div className="relative max-w-full max-h-full inline-block">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imgUrl(activeImg)}
                  alt=""
                  className="block max-w-full max-h-[78vh] object-contain select-none"
                  draggable={false}
                />
                {isIdeogram && showBoxes && (
                  <BoundingBoxEditor
                    boxes={editBoxes}
                    selectedIndex={selectedBoxIndex}
                    drawing={isDrawing}
                    onSelect={setSelectedBoxIndex}
                    onChangeBox={handleBoxChange}
                    onCreateBox={handleCreateBox}
                  />
                )}
              </div>
              {isIdeogram && (
                <button
                  type="button"
                  onClick={() => {
                    const next = !showBoxes;
                    setShowBoxes(next);
                    if (!next) {
                      setSelectedBoxIndex(null);
                      setIsDrawing(false);
                    }
                  }}
                  title={showBoxes ? 'Hide bounding boxes' : 'Show & edit bounding boxes'}
                  className={classNames('absolute top-2 right-2 bg-gray-900 rounded-full p-1 leading-[0px]', {
                    'text-blue-400': showBoxes,
                    'opacity-50 hover:opacity-100': !showBoxes,
                  })}
                >
                  <SquareDashed />
                </button>
              )}
            </div>

            <div className="bg-gray-950 w-full lg:w-96 shrink-0 flex flex-col gap-2 p-3 overflow-y-auto text-sm border-l border-gray-800">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={upsample}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-purple-500 bg-purple-600/20 text-purple-200 hover:bg-purple-600/30 text-xs transition-colors"
                  title="Upsample a description into a structured Ideogram caption"
                >
                  <Wand2 className="w-3.5 h-3.5" /> Upsample
                </button>
                <span className="text-[10px] text-gray-500 font-mono ml-auto">{captionExt}</span>
              </div>

              {isIdeogram ? (
                <IdeogramCaptionSidebar
                  caption={caption}
                  onChange={setCaption}
                  selectedIndex={selectedBoxIndex}
                  onSelectIndex={i => {
                    setSelectedBoxIndex(i);
                    if (i != null) setShowBoxes(true);
                  }}
                  isDrawing={isDrawing}
                  onToggleDrawing={() => setIsDrawing(d => !d)}
                  onSave={save}
                  isDirty={isDirty}
                />
              ) : (
                <div className="flex flex-col gap-2">
                  <textarea
                    className="w-full min-h-[12rem] rounded border-2 border-gray-700 bg-gray-900 text-gray-100 text-sm p-2 resize-none outline-none focus:border-blue-500"
                    placeholder="Plain caption — or convert to a structured Ideogram caption to place boxes."
                    value={caption}
                    onChange={e => setCaption(e.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setCaption(seedCaption(caption));
                      setShowBoxes(true);
                    }}
                    className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-purple-500 bg-purple-600/20 text-purple-200 hover:bg-purple-600/30 text-xs transition-colors"
                  >
                    <SquareDashed className="w-3.5 h-3.5" /> Convert to structured caption
                  </button>
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={save}
                      disabled={!isDirty}
                      className={classNames('px-4 py-1.5 rounded-md border text-xs font-medium transition-colors', {
                        'bg-green-600 border-green-500 text-white hover:bg-green-500': isDirty,
                        'border-gray-700 text-gray-500 cursor-default': !isDirty,
                      })}
                    >
                      {isDirty ? 'Save' : 'Saved'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </MainContent>

      <UpsamplePromptsModal />
    </>
  );
}
