'use client';
import { useMemo, useState } from 'react';
import {
  ModelArch,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
  groupPerceptualTrainingOptions,
} from './options';
import { useModelArchs } from '@/extensions/modelArchs';
import { defaultCompileOptions, defaultDatasetConfig } from './jobConfig';
import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { objectCopy, tagsToObj, objToTags } from '@/utils/basic';
import {
  TextInput,
  TextAreaInput,
  SelectInput,
  Checkbox,
  FormGroup,
  NumberInput,
  SliderInput,
  CreatableSelectInput,
} from '@/components/formInputs';
import Card from '@/components/Card';
import CustomTimestepCurvePicker from '@/components/CustomTimestepCurvePicker';
import { Plus, X, Copy, Wand2, SquareDashed, Info } from 'lucide-react';
import { openDoc } from '@/components/DocModal';
import { openUpsamplePromptsModal, toAspectRatio } from '@/components/UpsamplePromptsModal';
import { openPromptBoxEditor } from '@/components/PromptBoxEditorModal';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { IoFlaskSharp } from 'react-icons/io5';
import { QUICKSTARTS } from './quickstarts';
import { isMac } from '@/helpers/basic';

type Props = {
  jobConfig: JobConfig;
  // Optional key matches useNestedState's signature: when omitted the
  // entire state is replaced (used by the quickstart-template applier).
  setJobConfig: (value: any, key?: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  isLoading?: boolean;
};

const isDev = process.env.NODE_ENV === 'development';

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  isLoading,
}: Props) {
  // Quickstart selection is UI-only state — the chosen template is reflected
  // in the dropdown label but not stored in the saved config (the config IS
  // the template after apply; the dropdown is just a label for "which preset
  // shaped this form last").
  const [selectedQuickstart, setSelectedQuickstart] = useState<string>('custom');
  const { archs: modelArchs } = useModelArchs();
  const groupedModelOptions = useMemo(() => groupPerceptualTrainingOptions(modelArchs), [modelArchs]);
  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [modelArchs, jobConfig.config.process[0].model.arch]);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');
  const isAudioModel = !!(modelArch?.group === 'audio');
  // text-generating models: samples are media in, text out (no size)
  const isLlmModel = !!(modelArch?.group === 'llm');

  const taggedSampleArr: Record<string, any>[] | null = useMemo(() => {
    if (!modelArch) return null;
    if (!modelArch.sampleTags) return null;
    if (!jobConfig.config.process[0].sample.samples) return null;
    let sampleArr: any[] = [];
    for (let i = 0; i < jobConfig.config.process[0].sample.samples.length; i++) {
      const taggedPrompt = jobConfig.config.process[0].sample.samples[i].prompt;
      const tagsObj = tagsToObj(taggedPrompt);
      sampleArr.push(tagsObj);
    }
    return sampleArr;
  }, [modelArch, jobConfig.config.process[0].sample.samples]);

  const modelArchTagSections: SampleTags[] | null = useMemo(() => {
    if (!modelArch?.sampleTags) return null;
    const maxPerGroup = 5;
    let sections: SampleTags[] = [];
    let subSection: SampleTags = {};
    for (const [tagKey, tag] of Object.entries(modelArch.sampleTags)) {
      if ((tag.full && Object.keys(subSection).length > 0) || Object.keys(subSection).length >= maxPerGroup) {
        // reset the sub section build if the next tag is full or max per group is reached
        sections.push(subSection);
        subSection = {};
      }
      subSection[tagKey] = tag;
      if (tag.full) {
        // if the tag is full, push the section immediately and reset the sub section build
        sections.push(subSection);
        subSection = {};
      }
    }
    if (Object.keys(subSection).length > 0) {
      sections.push(subSection);
    }
    return sections.length > 0 ? sections : null;
  }, [modelArch]);

  const numTopCards = useMemo(() => {
    let count = 4; // job settings, model config, target config, save config
    if (modelArch?.additionalSections?.includes('model.multistage')) {
      count += 1; // add multistage card
    }
    if (!disableSections.includes('model.quantize')) {
      count += 1; // add quantization card
    }
    if (!disableSections.includes('slider')) {
      count += 1; // add slider card
    }
    return count;
  }, [modelArch, disableSections]);

  let topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-6';

  if (numTopCards == 5) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6';
  }
  if (numTopCards == 6) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-6';
  }

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    return count;
  }, [disableSections]);

  let trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  const showGPUSelect = !isMac();

  const validationConfig = jobConfig.config.process[0].train.validation_config;

  let numDatasetCols = 4;
  let numSampleTopCols = 4;
  let datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  let sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  if (isVideoModel) {
    numSampleTopCols += 1;
  }
  if (isAudioModel) {
    numDatasetCols -= 1;
    numSampleTopCols -= 1;
  }
  if (isLlmModel) {
    numSampleTopCols -= 1;
  }
  if (numDatasetCols == 3) {
    datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  if (numSampleTopCols == 5) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }
  if (numSampleTopCols == 3) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={`space-y-8 relative ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        {isLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-400 border-t-blue-500" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          </div>
        )}
        <div className={topBarClass}>
          <Card title="Job">
            <SelectInput
              label="Quickstart Template"
              docKey="config.quickstart"
              value={selectedQuickstart}
              onChange={(value: string) => {
                if (value === 'custom') {
                  setSelectedQuickstart('custom');
                  return;
                }
                const tmpl = QUICKSTARTS.find(q => q.id === value);
                if (!tmpl) return;
                const ok = window.confirm(
                  `Apply "${tmpl.label}" template?\n\nThis will overwrite the current config. ` +
                    `Your training name and dataset folder path will be preserved.`,
                );
                if (!ok) return;
                setJobConfig(tmpl.apply(jobConfig));
                setSelectedQuickstart(value);
              }}
              options={[
                { value: 'custom', label: 'Custom (no preset)' },
                ...QUICKSTARTS.map(q => ({ value: q.id, label: q.label })),
              ]}
            />
            <TextInput
              label="Training Name"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="Enter training name"
              disabled={runId !== null}
              required
            />
            {showGPUSelect && (
              <SelectInput
                label="GPU ID"
                value={`${gpuIDs}`}
                docKey="gpuids"
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            )}
            {disableSections.includes('trigger_word') ? null : (
              <TextInput
                label="Trigger Word"
                value={jobConfig.config.process[0].trigger_word || ''}
                docKey="config.process[0].trigger_word"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].trigger_word');
                }}
                placeholder=""
                required
              />
            )}
          </Card>

          {/* Model Configuration Section */}
          <Card title="Model">
            <SelectInput
              label="Model Architecture"
              value={jobConfig.config.process[0].model.arch}
              onChange={value => {
                handleModelArchChange(modelArchs, jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
              }}
              options={groupedModelOptions}
            />
            <TextInput
              label="Name or Path"
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].model.name_or_path');
              }}
              placeholder=""
              required
            />
            <FormGroup label="Base LoRA Merges" docKey="config.process[0].model.merge_loras">
              {(jobConfig.config.process[0].model.merge_loras ?? []).map((mergeLora, i) => (
                <div key={i} className="grid grid-cols-[minmax(0,1fr)_96px_32px] gap-2 items-end">
                  <TextInput
                    label={i === 0 ? 'Path' : undefined}
                    value={mergeLora.path ?? ''}
                    onChange={(value: string) =>
                      setJobConfig(value, `config.process[0].model.merge_loras[${i}].path`)
                    }
                    placeholder="/models/style.safetensors"
                  />
                  <NumberInput
                    label={i === 0 ? 'Weight' : undefined}
                    value={mergeLora.weight ?? 1}
                    onChange={value =>
                      setJobConfig(value ?? 1, `config.process[0].model.merge_loras[${i}].weight`)
                    }
                    placeholder="1.0"
                  />
                  <button
                    type="button"
                    title="Remove base LoRA"
                    className="h-8 w-8 flex items-center justify-center text-gray-400 hover:text-red-300"
                    onClick={() => {
                      const next = [...(jobConfig.config.process[0].model.merge_loras ?? [])];
                      next.splice(i, 1);
                      setJobConfig(next, 'config.process[0].model.merge_loras');
                    }}
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                type="button"
                className="mt-2 inline-flex items-center gap-2 text-xs text-gray-300 hover:text-white"
                onClick={() => {
                  const next = [...(jobConfig.config.process[0].model.merge_loras ?? [])];
                  next.push({ path: '', weight: 1.0 });
                  setJobConfig(next, 'config.process[0].model.merge_loras');
                }}
              >
                <Plus className="w-4 h-4" />
                Add LoRA
              </button>
            </FormGroup>
            {modelArch?.additionalSections?.includes('model.model_kwargs.instruction') && (
              <TextAreaInput
                label="LLM Prompt"
                className="pt-2"
                rows={2}
                docKey="model.model_kwargs.instruction"
                value={jobConfig.config.process[0].model.model_kwargs?.instruction ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.instruction')}
                placeholder="Describe this in detail."
                required
              />
            )}
            {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
              <TextInput
                label="Training Adapter Path"
                value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                docKey="config.process[0].model.assistant_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.additionalSections?.includes('model.unconditional_lora_path') && (
              <TextInput
                label="Unconditional Adapter Path"
                value={jobConfig.config.process[0].model.unconditional_lora_path ?? ''}
                docKey="config.process[0].model.unconditional_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.unconditional_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.customModelSelectOptions?.map(customOption =>
              customOption.type === 'checkbox' ? (
                <Checkbox
                  key={customOption.label}
                  label={customOption.label}
                  checked={customOption.getValue(jobConfig)}
                  doc={customOption.doc}
                  onChange={value => customOption.onChange(value, jobConfig, setJobConfig)}
                />
              ) : (
                <SelectInput
                  key={customOption.label}
                  label={customOption.label}
                  value={customOption.getValue(jobConfig) ?? ''}
                  doc={customOption.doc}
                  onChange={value => customOption.onChange(value, jobConfig, setJobConfig)}
                  options={customOption.options}
                />
              ),
            )}
            {modelArch?.modelNotes && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: `Notes - ${modelArch.label}`,
                      description: <div className="space-y-3">{modelArch.modelNotes}</div>,
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-blue-950/60 border border-blue-800 px-3 py-2 text-sm text-blue-200 hover:bg-blue-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-blue-400" />
                  <span>Model notes</span>
                </button>
              </div>
            )}
            {modelArch?.gateUrl && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: 'Gated Model',
                      description: (
                        <div className="space-y-3">
                          <p>
                            This model is gated on Huggingface. Before you can use it, you will need to accept the model
                            terms on the model page:
                          </p>
                          <p>
                            <a
                              href={gateUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              {gateUrl}
                            </a>
                          </p>
                          <p>
                            You will also need to create a Huggingface{' '}
                            <a
                              href="https://huggingface.co/settings/tokens"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              read token
                            </a>{' '}
                            and add it on the{' '}
                            <a href="/settings" className="text-blue-400 hover:text-blue-300 underline">
                              settings page
                            </a>
                            .
                          </p>
                        </div>
                      ),
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-yellow-950/60 border border-yellow-800 px-3 py-2 text-sm text-yellow-200 hover:bg-yellow-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-yellow-400" />
                  <span>
                    Gated model. <span className="underline">Learn more.</span>
                  </span>
                </button>
              </div>
            )}
            {modelArch?.additionalSections?.includes('model.low_vram') && (
              <FormGroup label="Options">
                <Checkbox
                  label="Low VRAM"
                  checked={jobConfig.config.process[0].model.low_vram}
                  onChange={value => setJobConfig(value, 'config.process[0].model.low_vram')}
                />
              </FormGroup>
            )}
            {modelArch?.additionalSections?.includes('model.model_kwargs.kv_cache') && (
              <Checkbox
                label="KV Cache"
                docKey="model.model_kwargs.kv_cache"
                checked={jobConfig.config.process[0].model.model_kwargs.kv_cache || false}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.kv_cache')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
              <Checkbox
                label="Match Target Res"
                docKey="model.qie.match_target_res"
                checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.layer_offloading') && !isMac() && (
              <>
                <Checkbox
                  label={
                    <>
                      Layer Offloading <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                    </>
                  }
                  checked={jobConfig.config.process[0].model.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading')}
                  docKey="model.layer_offloading"
                />
                {jobConfig.config.process[0].model.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Transformer Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                    <SliderInput
                      label="Text Encoder Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </Card>
          {disableSections.includes('model.quantize') ? null : (
            <Card title="Quantize / Compile">
              <SelectInput
                label="Transformer"
                value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype');
                }}
                options={transformerQuantizationOptions}
              />
              {!disableSections.includes('model.quantize_te') && !isLlmModel && (
                <SelectInput
                  label="Text Encoder"
                  value={
                    jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''
                  }
                  onChange={value => {
                    if (value === '') {
                      setJobConfig(false, 'config.process[0].model.quantize_te');
                      value = defaultQtype;
                    } else {
                      setJobConfig(true, 'config.process[0].model.quantize_te');
                    }
                    setJobConfig(value, 'config.process[0].model.qtype_te');
                  }}
                  options={quantizationOptions}
                />
              )}
              {!isLlmModel && (
                <>
                  <FormGroup label="Compile Options">
                    <></>
                  </FormGroup>
                  <Checkbox
                    label="Compile Model"
                    checked={jobConfig.config.process[0].model.compile || false}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].model.compile');
                      if (value) {
                        for (const key in defaultCompileOptions) {
                          setJobConfig((defaultCompileOptions as any)[key], `config.process[0].model.${key}`);
                        }
                      } else {
                        for (const key in defaultCompileOptions) {
                          setJobConfig(undefined, `config.process[0].model.${key}`);
                        }
                      }
                    }}
                  />
                </>
              )}
            </Card>
          )}
          {modelArch?.additionalSections?.includes('model.multistage') && (
            <Card title="Multistage">
              <FormGroup label="Stages to Train" docKey={'model.multistage'}>
                <Checkbox
                  label="High Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                />
                <Checkbox
                  label="Low Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                />
              </FormGroup>
              <NumberInput
                label="Switch Every"
                value={jobConfig.config.process[0].train.switch_boundary_every}
                onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                placeholder="eg. 1"
                docKey={'train.switch_boundary_every'}
                min={1}
                required
              />
            </Card>
          )}
          <Card title="Target">
            <SelectInput
              label="Target Type"
              value={jobConfig.config.process[0].network?.type ?? 'lora'}
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={[
                { value: 'lora', label: 'LoRA' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {jobConfig.config.process[0].network?.type == 'lokr' && (
              <SelectInput
                label="LoKr Factor"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <>
                <NumberInput
                  label="Linear Rank"
                  value={jobConfig.config.process[0].network.linear}
                  onChange={value => {
                    console.log('onChange', value);
                    setJobConfig(value, 'config.process[0].network.linear');
                    setJobConfig(value, 'config.process[0].network.linear_alpha');
                  }}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                />
                {disableSections.includes('network.conv') ? null : (
                  <NumberInput
                    label="Conv Rank"
                    value={jobConfig.config.process[0].network.conv}
                    onChange={value => {
                      console.log('onChange', value);
                      setJobConfig(value, 'config.process[0].network.conv');
                      setJobConfig(value, 'config.process[0].network.conv_alpha');
                    }}
                    placeholder="eg. 16"
                    min={0}
                    max={1024}
                  />
                )}
              </>
            )}
          </Card>
          {!disableSections.includes('slider') && (
            <Card title="Slider">
              <TextInput
                label="Target Class"
                className=""
                value={jobConfig.config.process[0].slider?.target_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                placeholder="eg. person"
              />
              <TextInput
                label="Positive Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                placeholder="eg. person who is happy"
              />
              <TextInput
                label="Negative Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                placeholder="eg. person who is sad"
              />
              <TextInput
                label="Anchor Class"
                className=""
                value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                placeholder=""
              />
            </Card>
          )}
          <Card title="Save">
            <SelectInput
              label="Data Type"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="Save Every"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="Max Step Saves to Keep"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
            <Checkbox
              label="Save optimizer per checkpoint"
              checked={jobConfig.config.process[0].save.save_optimizer_per_checkpoint || false}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_optimizer_per_checkpoint')}
            />
          </Card>
        </div>
        <div>
          <Card title="Training">
            <div className={trainingBarClass}>
              <div>
                <NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                <NumberInput
                  label="Steps"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                />
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'automagic', label: 'Automagic' },
                    { value: 'automagic2', label: 'Automagic v2' },
                    { value: 'automagic3', label: 'Automagic v3' },
                    { value: 'rose', label: 'Rose (stateless, experimental)' },
                    { value: 'prodigyopt', label: 'Prodigy' },
                    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
                  ]}
                />
                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
              </div>
              {!isLlmModel && (
                <div>
                  {disableSections.includes('train.timestep_type') ? null : (
                    <SelectInput
                      label="Timestep Type"
                      value={jobConfig.config.process[0].train.timestep_type}
                      disabled={disableSections.includes('train.timestep_type') || false}
                      onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid' },
                      { value: 'linear', label: 'Linear' },
                      { value: 'shift', label: 'Shift' },
                      { value: 'weighted', label: 'Weighted' },
                      { value: 'weighted_low', label: 'Weighted Low' },
                      { value: 'custom', label: 'Custom Weighting Curve' },
                    ]}
                  />
                )}
                {jobConfig.config.process[0].train.timestep_type === 'custom' &&
                 !disableSections.includes('train.timestep_type') && (
                  <CustomTimestepCurvePicker
                    value={jobConfig.config.process[0].train.custom_timestep_curve ?? null}
                    onChange={next => setJobConfig(next, 'config.process[0].train.custom_timestep_curve')}
                    apiBase="/api/timestep-curves"
                    manageHref="/timestep-curves"
                    label="Custom Weighting Curve"
                  />
                )}
                <SelectInput
                  label="Timestep Bias"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'High Noise' },
                    { value: 'style', label: 'Low Noise' },
                    { value: 'custom', label: 'Custom Distribution' },
                  ]}
                />
                {jobConfig.config.process[0].train.content_or_style === 'custom' && (
                  <CustomTimestepCurvePicker
                    value={jobConfig.config.process[0].train.custom_timestep_distribution ?? null}
                    onChange={next => setJobConfig(next, 'config.process[0].train.custom_timestep_distribution')}
                    apiBase="/api/timestep-distributions"
                    manageHref="/timestep-distributions"
                    label="Custom Distribution"
                  />
                )}
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                  ]}
                />
                <NumberInput
                  label="Diffusion Loss Weight"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.diffusion_loss_weight ?? 1.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].train.diffusion_loss_weight')
                  }
                  placeholder="1.0 = normal"
                  min={0}
                  max={1}
                />
                {(jobConfig.config.process[0].train.diffusion_loss_weight ?? 1.0) > 0 && (
                  <>
                    <NumberInput
                      label="Diffusion Loss Min t"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.diffusion_loss_min_t ?? 0.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.diffusion_loss_min_t')
                      }
                      placeholder="0.0 = all timesteps"
                      min={0}
                      max={1}
                    />
                    <NumberInput
                      label="Diffusion Loss Max t"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.diffusion_loss_max_t ?? 1.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.diffusion_loss_max_t')
                      }
                      placeholder="1.0 = all timesteps"
                      min={0}
                      max={1}
                    />
                  </>
                )}
                <NumberInput
                  label="Min Denoising Step"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.min_denoising_steps ?? 0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].train.min_denoising_steps')
                  }
                  placeholder="0"
                  min={0}
                  max={999}
                />
                <NumberInput
                  label="Max Denoising Step"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.max_denoising_steps ?? 999}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].train.max_denoising_steps')
                  }
                  placeholder="999"
                  min={0}
                  max={999}
                />
                {/* Global default for per-dataset loss_split.
                    'auto' = key omitted: trainer turns it on for any dataset
                            whose effective depth-consistency weight is > 0.
                    'diffusion_depth' = force on everywhere (string).
                    'off' = explicit null in YAML: force off everywhere. */}
                <SelectInput
                  label="Loss Split (global)"
                  className="pt-2"
                  value={
                    jobConfig.config.process[0].train.loss_split === undefined
                      ? 'auto'
                      : jobConfig.config.process[0].train.loss_split === null
                      ? 'off'
                      : jobConfig.config.process[0].train.loss_split
                  }
                  onChange={value => {
                    if (value === 'auto') {
                      setJobConfig(undefined, 'config.process[0].train.loss_split');
                    } else if (value === 'off') {
                      setJobConfig(null, 'config.process[0].train.loss_split');
                    } else {
                      setJobConfig(value, 'config.process[0].train.loss_split');
                    }
                  }}
                  options={[
                    { value: 'auto', label: 'Auto (on when depth anchor is active)' },
                    { value: 'diffusion_depth', label: 'Force on (Diffusion / Depth alternating)' },
                    { value: 'off', label: 'Force off (sum every step)' },
                  ]}
                />
                {/* Latent perceptual loss — experimental, hidden for now
                <NumberInput
                  label="Latent Perceptual Loss Weight"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.latent_perceptual_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].train.latent_perceptual_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].train.latent_perceptual_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Latent Perceptual Min t"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.latent_perceptual_loss_min_t ?? 0.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.latent_perceptual_loss_min_t')
                      }
                      placeholder="0.0"
                      min={0}
                      max={1}
                    />
                    <NumberInput
                      label="Latent Perceptual Max t"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.latent_perceptual_loss_max_t ?? 0.5}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.latent_perceptual_loss_max_t')
                      }
                      placeholder="0.5"
                      min={0}
                      max={1}
                    />
                  </>
                )}
                */}
                  {modelArch?.additionalSections?.includes('train.audio_loss_multiplier') && (
                    <NumberInput
                      label="Audio Loss Multiplier"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.audio_loss_multiplier ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.audio_loss_multiplier')}
                      placeholder="eg. 1.0"
                      docKey={'train.audio_loss_multiplier'}
                      min={0}
                    />
                  )}
                </div>
              )}
              <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="Weight Noise" className="pt-2" docKey={'train.weight_noise'}>
                  <Checkbox
                    label="Enable Weight Noise"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.weight_noise?.enabled || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.weight_noise.enabled')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.weight_noise?.enabled && (
                  <>
                    <SelectInput
                      label="Mode"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.weight_noise?.mode || 'relative'}
                      onChange={value => setJobConfig(value, 'config.process[0].train.weight_noise.mode')}
                      options={[
                        { value: 'relative', label: 'Relative (σ × per-param weight RMS)' },
                        { value: 'absolute', label: 'Absolute (fixed σ)' },
                      ]}
                    />
                    <NumberInput
                      label="Sigma"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.weight_noise?.sigma as number}
                      onChange={value => setJobConfig(value, 'config.process[0].train.weight_noise.sigma')}
                      placeholder="0.001 – 0.0017"
                      min={0}
                      docKey={'train.weight_noise.sigma'}
                    />
                    <NumberInput
                      label="Log Every"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.weight_noise?.log_every as number}
                      onChange={value => setJobConfig(value, 'config.process[0].train.weight_noise.log_every')}
                      placeholder="eg. 50"
                      min={0}
                    />
                  </>
                )}

                {!isLlmModel && (
                  <FormGroup label="Text Encoder Optimizations" className="pt-2">
                    {!disableSections.includes('train.unload_text_encoder') && (
                      <Checkbox
                        label="Unload TE"
                        checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                        docKey={'train.unload_text_encoder'}
                        onChange={value => {
                          setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                          if (value) {
                            setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                          }
                        }}
                      />
                    )}
                    <Checkbox
                      label="Cache Text Embeddings"
                      checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                      docKey={'train.cache_text_embeddings'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                        }
                      }}
                    />
                  </FormGroup>
                )}
              </div>
              <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="Regularization">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Differential Output Preservation"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP Loss Multiplier"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP Preservation Class"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Blank Prompt Preservation"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP Loss Multiplier"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
                <FormGroup label="Perceptual Anchoring" docKey="perceptual_anchoring">
                  <></>
                </FormGroup>
                {/* Face ID Conditioning — hidden for now, available via YAML config
                <Checkbox
                  label="Face ID Conditioning"
                  docKey="face_id.enabled"
                  className="pt-1"
                  checked={jobConfig.config.process[0].face_id?.enabled || false}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].face_id.enabled');
                  }}
                />
                {jobConfig.config.process[0].face_id?.enabled && (
                  <>
                    <NumberInput
                      label="Face Tokens"
                      docKey="face_id.num_tokens"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.num_tokens || 4}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.num_tokens')
                      }
                      placeholder="eg. 4"
                      min={1}
                      max={16}
                    />
                    <NumberInput
                      label="Face Dropout"
                      docKey="face_id.dropout_prob"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.dropout_prob || 0.1}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.dropout_prob')
                      }
                      placeholder="eg. 0.1"
                      min={0}
                      max={1}
                    />
                    <NumberInput
                      label="Scale LR Multiplier"
                      docKey="face_id.scale_lr_multiplier"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.scale_lr_multiplier || 10}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.scale_lr_multiplier')
                      }
                      placeholder="eg. 10"
                      min={1}
                      max={100}
                    />
                    <NumberInput
                      label="Init Scale"
                      docKey="face_id.init_scale"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.init_scale || 0.01}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.init_scale')
                      }
                      placeholder="eg. 0.01"
                      min={0.001}
                      max={1000}
                    />
                    <Checkbox
                      label="Vision Face Embeddings (CLIP/DINOv2)"
                      docKey="face_id.vision_enabled"
                      className="pt-3"
                      checked={jobConfig.config.process[0].face_id?.vision_enabled || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].face_id.vision_enabled');
                      }}
                    />
                    {jobConfig.config.process[0].face_id?.vision_enabled && (
                      <>
                        <SelectInput
                          label="Vision Model"
                          className="pt-2"
                          value={jobConfig.config.process[0].face_id?.vision_model || 'openai/clip-vit-large-patch14'}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].face_id.vision_model')
                          }
                          options={[
                            { label: 'CLIP ViT-L/14', value: 'openai/clip-vit-large-patch14' },
                            { label: 'DINOv2 Large', value: 'facebook/dinov2-large' },
                          ]}
                        />
                        <NumberInput
                          label="Vision Tokens"
                          className="pt-2"
                          value={jobConfig.config.process[0].face_id?.vision_num_tokens || 4}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].face_id.vision_num_tokens')
                          }
                          placeholder="eg. 4"
                          min={1}
                          max={16}
                        />
                      </>
                    )}
                  </>
                )}
                */}
                <Checkbox
                  label="Identity Metrics (track without loss)"
                  docKey="face_id.identity_metrics"
                  className="pt-3"
                  checked={jobConfig.config.process[0].face_id?.identity_metrics || false}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.identity_metrics')
                  }
                />
                <NumberInput
                  label="Identity Loss Weight"
                  docKey="face_id.identity_loss_weight"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.identity_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.identity_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].face_id?.identity_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Identity Loss Min t"
                      docKey="face_id.identity_loss_min_t"
                      className="pt-2"
                      value={
                        jobConfig.config.process[0].face_id?.identity_loss_min_t ?? 0.0
                      }
                      onChange={value =>
                        setJobConfig(
                          value,
                          'config.process[0].face_id.identity_loss_min_t'
                        )
                      }
                      placeholder="eg. 0"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Identity Loss Max t"
                      docKey="face_id.identity_loss_max_t"
                      className="pt-2"
                      value={
                        jobConfig.config.process[0].face_id?.identity_loss_max_t ?? 1.0
                      }
                      onChange={value =>
                        setJobConfig(
                          value,
                          'config.process[0].face_id.identity_loss_max_t'
                        )
                      }
                      placeholder="eg. 1"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Identity Loss Min Cosine"
                      docKey="face_id.identity_loss_min_cos"
                      className="pt-2"
                      value={
                        jobConfig.config.process[0].face_id?.identity_loss_min_cos ?? 0.2
                      }
                      onChange={value =>
                        setJobConfig(
                          value,
                          'config.process[0].face_id.identity_loss_min_cos'
                        )
                      }
                      placeholder="0.2 = only apply when face detected"
                      min={0.0}
                      max={1.0}
                    />
                    <Checkbox
                      label="Use Average Face Embedding"
                      docKey="face_id.identity_loss_use_average"
                      className="pt-2"
                      checked={jobConfig.config.process[0].face_id?.identity_loss_use_average || false}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.identity_loss_use_average')
                      }
                    />
                    <NumberInput
                      label="Average Blend (0=per-image, 0.5=midpoint, 1=average)"
                      docKey="face_id.identity_loss_average_blend"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.identity_loss_average_blend ?? 0.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.identity_loss_average_blend')
                      }
                      placeholder="0 = per-image only"
                      min={0}
                      max={1}
                    />
                    <Checkbox
                      label="Use Random Face Embedding Per Step"
                      docKey="face_id.identity_loss_use_random"
                      className="pt-2"
                      checked={jobConfig.config.process[0].face_id?.identity_loss_use_random || false}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.identity_loss_use_random')
                      }
                    />
                    <NumberInput
                      label="Multi-Ref Count (0 = disabled)"
                      docKey="face_id.identity_loss_num_refs"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.identity_loss_num_refs ?? 0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.identity_loss_num_refs')
                      }
                      placeholder="0 = use single ref"
                      min={0}
                    />
                  </>
                )}
                {/* Landmark loss — experimental, hidden for now
                <NumberInput
                  label="Landmark Loss Weight"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.landmark_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.landmark_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                */}
                <NumberInput
                  label="Body Proportion Loss Weight"
                  docKey="face_id.body_proportion_loss_weight"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.body_proportion_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.body_proportion_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].face_id?.body_proportion_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Body Proportion Min t"
                      docKey="face_id.body_proportion_loss_min_t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.body_proportion_loss_min_t ?? 0.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.body_proportion_loss_min_t')
                      }
                      placeholder="eg. 0"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Body Proportion Max t"
                      docKey="face_id.body_proportion_loss_max_t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.body_proportion_loss_max_t ?? 1.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.body_proportion_loss_max_t')
                      }
                      placeholder="eg. 1"
                      min={0.0}
                      max={1.0}
                    />
                  </>
                )}
                <NumberInput
                  label="Face Suppression Weight"
                  docKey="face_id.face_suppression_weight"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.face_suppression_weight ?? null}
                  onChange={value =>
                    setJobConfig(value === null || value === undefined ? undefined : value, 'config.process[0].face_id.face_suppression_weight')
                  }
                  placeholder="none (no suppression)"
                  min={0}
                  max={1}
                />
                {(jobConfig.config.process[0].face_id?.face_suppression_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Suppression Expand"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.face_suppression_expand ?? 2.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.face_suppression_expand')
                      }
                      placeholder="1.0 = face only, 1.8 = full head"
                      min={1.0}
                      max={3.0}
                    />
                    <Checkbox
                      label="Soft Gaussian Falloff"
                      className="pt-2"
                      checked={jobConfig.config.process[0].face_id?.face_suppression_soft ?? false}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.face_suppression_soft')
                      }
                    />
                  </>
                )}
                {/* -------- Subject Masking (YOLO + SAM 2 + SegFormer) -------- */}
                <Checkbox
                  label="Subject Masking (auto body/clothing masks)"
                  docKey="subject_mask.enabled"
                  className="pt-4"
                  checked={jobConfig.config.process[0].subject_mask?.enabled || false}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].subject_mask.enabled')
                  }
                />
                {jobConfig.config.process[0].subject_mask?.enabled && (
                  <>
                    <SelectInput
                      label="SAM 2 Size"
                      docKey="subject_mask.sam_size"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.sam_size ?? 'small'}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.sam_size')
                      }
                      options={[
                        { label: 'tiny (31M params)', value: 'tiny' },
                        { label: 'small (39M params)', value: 'small' },
                        { label: 'base_plus (73M params)', value: 'base_plus' },
                        { label: 'large (217M params)', value: 'large' },
                      ]}
                    />
                    <NumberInput
                      label="YOLO Confidence Threshold"
                      docKey="subject_mask.yolo_conf"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.yolo_conf ?? 0.25}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.yolo_conf')
                      }
                      placeholder="0.25"
                      min={0.05}
                      max={0.95}
                    />
                    <Checkbox
                      label="Primary Person Only"
                      docKey="subject_mask.primary_only"
                      className="pt-2"
                      checked={jobConfig.config.process[0].subject_mask?.primary_only ?? true}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.primary_only')
                      }
                    />
                    <NumberInput
                      label="SegFormer Resolution"
                      docKey="subject_mask.segformer_res"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.segformer_res ?? 768}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.segformer_res')
                      }
                      placeholder="768"
                      min={256}
                      max={1536}
                    />
                    <NumberInput
                      label="Cache Resolution"
                      docKey="subject_mask.cache_resolution"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.cache_resolution ?? 256}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.cache_resolution')
                      }
                      placeholder="256"
                      min={64}
                      max={1024}
                    />
                    <NumberInput
                      label="Body Close Radius"
                      docKey="subject_mask.body_close_radius"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.body_close_radius ?? 2}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.body_close_radius')
                      }
                      placeholder="2 (higher fills blotchy gaps; re-extracts cache)"
                      min={0}
                      max={12}
                    />
                    <NumberInput
                      label="Background Loss Weight"
                      docKey="subject_mask.background_loss_weight"
                      className="pt-3"
                      value={jobConfig.config.process[0].subject_mask?.background_loss_weight ?? null}
                      onChange={value =>
                        setJobConfig(value === null || value === undefined ? undefined : value,
                          'config.process[0].subject_mask.background_loss_weight')
                      }
                      placeholder="none (no change); 0 = ignore background"
                      min={0}
                    />
                    <NumberInput
                      label="Clothing Loss Weight"
                      docKey="subject_mask.clothing_loss_weight"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.clothing_loss_weight ?? null}
                      onChange={value =>
                        setJobConfig(value === null || value === undefined ? undefined : value,
                          'config.process[0].subject_mask.clothing_loss_weight')
                      }
                      placeholder="none (no change); <1 = de-emphasize"
                      min={0}
                    />
                    <NumberInput
                      label="Body Loss Weight"
                      docKey="subject_mask.body_loss_weight"
                      className="pt-2"
                      value={jobConfig.config.process[0].subject_mask?.body_loss_weight ?? null}
                      onChange={value =>
                        setJobConfig(value === null || value === undefined ? undefined : value,
                          'config.process[0].subject_mask.body_loss_weight')
                      }
                      placeholder="none (no change); >1 = boost body"
                      min={0}
                    />
                    <Checkbox
                      label="Restrict Perceptual Losses to Body"
                      docKey="subject_mask.perceptual_restrict_to_body"
                      className="pt-2"
                      checked={jobConfig.config.process[0].subject_mask?.perceptual_restrict_to_body ?? false}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.perceptual_restrict_to_body')
                      }
                    />
                    <Checkbox
                      label="Save Debug Preview Tiles"
                      docKey="subject_mask.save_debug_previews"
                      className="pt-2"
                      checked={jobConfig.config.process[0].subject_mask?.save_debug_previews ?? false}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].subject_mask.save_debug_previews')
                      }
                    />
                  </>
                )}
                {/* -------- Depth Consistency (Depth-Anything-V2 SSI + gradient) -------- */}
                <NumberInput
                  label="Depth Consistency Loss Weight"
                  docKey="depth_consistency.loss_weight"
                  className="pt-4"
                  value={jobConfig.config.process[0].depth_consistency?.loss_weight ?? 0.1}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].depth_consistency.loss_weight')
                  }
                  placeholder="0.1 (Small) / 0.001 (Large); 0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].depth_consistency?.loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Depth Min t"
                      docKey="depth_consistency.loss_min_t"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.loss_min_t ?? 0.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.loss_min_t')
                      }
                      placeholder="eg. 0"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Depth Max t"
                      docKey="depth_consistency.loss_max_t"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.loss_max_t ?? 1.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.loss_max_t')
                      }
                      placeholder="eg. 0.9"
                      min={0.0}
                      max={1.0}
                    />
                    <SelectInput
                      label="Depth Mask Source"
                      docKey="depth_consistency.mask_source"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.mask_source ?? 'subject'}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.mask_source')
                      }
                      options={[
                        { label: 'none (full image)', value: 'none' },
                        { label: 'subject (person mask)', value: 'subject' },
                        { label: 'body (identity-relevant only)', value: 'body' },
                      ]}
                    />
                    <NumberInput
                      label="SSI L1 Weight"
                      docKey="depth_consistency.ssi_weight"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.ssi_weight ?? 1.0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.ssi_weight')
                      }
                      placeholder="eg. 1.0"
                      min={0}
                    />
                    <NumberInput
                      label="Gradient Matching Weight"
                      docKey="depth_consistency.grad_weight"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.grad_weight ?? 0.5}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.grad_weight')
                      }
                      placeholder="eg. 0.5"
                      min={0}
                    />
                    <NumberInput
                      label="Gradient Scales"
                      docKey="depth_consistency.grad_scales"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.grad_scales ?? 4}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.grad_scales')
                      }
                      placeholder="eg. 4"
                      min={1}
                      max={8}
                    />
                    <NumberInput
                      label="Preview Every (steps)"
                      docKey="depth_consistency.preview_every"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.preview_every ?? 100}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.preview_every')
                      }
                      placeholder="eg. 100; 0 disables"
                      min={0}
                    />
                    <TextInput
                      label="DA2 Model ID"
                      docKey="depth_consistency.model_id"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.model_id ?? 'depth-anything/Depth-Anything-V2-Small-hf'}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.model_id')
                      }
                      placeholder="depth-anything/Depth-Anything-V2-Small-hf"
                    />
                    <NumberInput
                      label="DA2 Input Size"
                      docKey="depth_consistency.input_size"
                      className="pt-2"
                      value={jobConfig.config.process[0].depth_consistency?.input_size ?? 518}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.input_size')
                      }
                      placeholder="518 (multiple of 14)"
                      min={224}
                      max={1400}
                    />
                    <Checkbox
                      label="Gradient Checkpointing (DA2)"
                      docKey="depth_consistency.grad_checkpoint"
                      className="pt-2"
                      checked={jobConfig.config.process[0].depth_consistency?.grad_checkpoint ?? true}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].depth_consistency.grad_checkpoint')
                      }
                    />
                  </>
                )}
                {/* Body shape, normal map, VAE anchor, body conditioning — experimental, hidden for now
                <NumberInput
                  label="Body Shape Loss Weight (HybrIK)"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.body_shape_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.body_shape_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].face_id?.body_shape_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Body Shape Min t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.body_shape_loss_min_t ?? 0.4}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.body_shape_loss_min_t')
                      }
                      placeholder="eg. 0.4"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Body Shape Max t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.body_shape_loss_max_t ?? 0.8}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.body_shape_loss_max_t')
                      }
                      placeholder="eg. 0.8"
                      min={0.0}
                      max={1.0}
                    />
                  </>
                )}
                <NumberInput
                  label="Normal Map Loss Weight (Sapiens)"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.normal_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.normal_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].face_id?.normal_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="Normal Loss Min t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.normal_loss_min_t ?? 0.4}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.normal_loss_min_t')
                      }
                      placeholder="eg. 0.4"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="Normal Loss Max t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.normal_loss_max_t ?? 0.8}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.normal_loss_max_t')
                      }
                      placeholder="eg. 0.8"
                      min={0.0}
                      max={1.0}
                    />
                  </>
                )}
                <NumberInput
                  label="VAE Anchor Loss Weight"
                  className="pt-3"
                  value={jobConfig.config.process[0].face_id?.vae_anchor_loss_weight ?? 0.0}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].face_id.vae_anchor_loss_weight')
                  }
                  placeholder="0 = disabled"
                  min={0}
                />
                {(jobConfig.config.process[0].face_id?.vae_anchor_loss_weight ?? 0) > 0 && (
                  <>
                    <NumberInput
                      label="VAE Anchor Min t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.vae_anchor_loss_min_t ?? 0}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.vae_anchor_loss_min_t')
                      }
                      placeholder="eg. 0"
                      min={0.0}
                      max={1.0}
                    />
                    <NumberInput
                      label="VAE Anchor Max t"
                      className="pt-2"
                      value={jobConfig.config.process[0].face_id?.vae_anchor_loss_max_t ?? 0.5}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].face_id.vae_anchor_loss_max_t')
                      }
                      placeholder="eg. 0.5"
                      min={0.0}
                      max={1.0}
                    />
                  </>
                )}
                */}
                {/* Body Shape Conditioning (SMPL) — experimental, hidden for now
                <Checkbox
                  label="Body Shape Conditioning (SMPL)"
                  className="pt-3"
                  checked={jobConfig.config.process[0].body_id?.enabled || false}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].body_id.enabled');
                  }}
                />
                {jobConfig.config.process[0].body_id?.enabled && (
                  <>
                    <NumberInput
                      label="Body Tokens"
                      className="pt-2"
                      value={jobConfig.config.process[0].body_id?.num_tokens || 4}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].body_id.num_tokens')
                      }
                      placeholder="eg. 4"
                      min={1}
                      max={16}
                    />
                    <NumberInput
                      label="Body Dropout"
                      className="pt-2"
                      value={jobConfig.config.process[0].body_id?.dropout_prob || 0.1}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].body_id.dropout_prob')
                      }
                      placeholder="eg. 0.1"
                      min={0}
                      max={1}
                    />
                    <NumberInput
                      label="Scale LR Multiplier"
                      className="pt-2"
                      value={jobConfig.config.process[0].body_id?.scale_lr_multiplier || 10}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].body_id.scale_lr_multiplier')
                      }
                      placeholder="eg. 10"
                      min={1}
                      max={100}
                    />
                    <NumberInput
                      label="Init Scale"
                      className="pt-2"
                      value={jobConfig.config.process[0].body_id?.init_scale || 0.01}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].body_id.init_scale')
                      }
                      placeholder="eg. 0.01"
                      min={0.001}
                      max={1000}
                    />
                  </>
                )}
                */}
              </div>
            </div>
          </Card>
        </div>
        <div>
          {!isLlmModel && (
            <Card
              title="Validation"
              toggled={!!validationConfig}
              onToggle={value => {
                if (value) {
                  setJobConfig(
                    {
                      validation_items: [{ image_path: '', prompt: '' }],
                      resolution: 1024,
                      validate_every_n_steps: 1,
                      validation_sigmas: [0.5],
                    },
                    'config.process[0].train.validation_config',
                  );
                } else {
                  setJobConfig(undefined, 'config.process[0].train.validation_config');
                }
              }}
            >
              {validationConfig && (
                <>
                  <p className="text-sm text-gray-400 mb-4">
                    Validation runs a stable loss check on a fixed set of images. Each image is encoded once at startup
                    and predicted at the selected sigmas with fixed seeds, so the result is always deterministic and
                    comparable across the run. The average loss is logged as val/loss every time validation runs. The
                    images need to match the concept of your dataset, but{' '}
                    <span className="font-bold text-gray-300">do not include the validation images in the dataset</span>
                    . They must be images containing the concept you want to train, but not an image trained on.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <NumberInput
                      label="Validate Every"
                      value={validationConfig.validate_every_n_steps}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.validation_config.validate_every_n_steps')
                      }
                      placeholder="eg. 10"
                      min={1}
                      required
                    />
                    <NumberInput
                      label="Validation Resolution"
                      value={validationConfig.resolution}
                      onChange={value => setJobConfig(value, 'config.process[0].train.validation_config.resolution')}
                      placeholder="eg. 512"
                      min={64}
                      required
                    />
                    <SelectInput
                      label="Validation Sigmas"
                      value={(validationConfig.validation_sigmas ?? [1.0, 0.75, 0.5, 0.25]).join(', ')}
                      onChange={value =>
                        setJobConfig(
                          value.split(',').map((v: string) => parseFloat(v)),
                          'config.process[0].train.validation_config.validation_sigmas',
                        )
                      }
                      options={[
                        { value: '0.5', label: '0.5' },
                        { value: '1, 0.5', label: '1.0, 0.5' },
                        { value: '1, 0.66, 0.33', label: '1.0, 0.66, 0.33' },
                        { value: '1, 0.75, 0.5, 0.25', label: '1.0, 0.75, 0.5, 0.25' },
                      ]}
                    />
                  </div>
                  <div className="mt-4">
                    <label className="block text-xs text-gray-300 mb-2">
                      Validation Images ({validationConfig.validation_items.length})
                    </label>
                    {validationConfig.validation_items.map((item, i) => (
                      <div key={i} className="rounded-lg pl-4 pr-1 py-3 mb-4 bg-gray-950">
                        <div className="flex items-center space-x-4">
                          <SampleControlImage
                            instruction="Add Image"
                            src={item.image_path === '' ? null : item.image_path}
                            onNewImageSelected={imagePath => {
                              setJobConfig(
                                imagePath ?? '',
                                `config.process[0].train.validation_config.validation_items[${i}].image_path`,
                              );
                            }}
                          />
                          <div className="flex-1">
                            <TextInput
                              label="Prompt"
                              value={item.prompt}
                              onChange={value =>
                                setJobConfig(
                                  value,
                                  `config.process[0].train.validation_config.validation_items[${i}].prompt`,
                                )
                              }
                              placeholder="Enter prompt"
                            />
                          </div>
                          <div>
                            <button
                              type="button"
                              onClick={() =>
                                setJobConfig(
                                  validationConfig.validation_items.filter((_, index) => index !== i),
                                  'config.process[0].train.validation_config.validation_items',
                                )
                              }
                              className="rounded-full p-1 text-sm"
                            >
                              <X />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          [...validationConfig.validation_items, { image_path: '', prompt: '' }],
                          'config.process[0].train.validation_config.validation_items',
                        )
                      }
                      className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                    >
                      Add Validation Image
                    </button>
                  </div>
                </>
              )}
            </Card>
          )}
        </div>
        <div>
          {!isLlmModel && (
            <Card title="Advanced" collapsible>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div>
                  <Checkbox
                    label="Do Differential Guidance"
                    docKey={'train.do_differential_guidance'}
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                    onChange={value => {
                      let newValue = value == false ? undefined : value;
                      setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                      if (!newValue) {
                        setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                      } else if (
                        jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                        jobConfig.config.process[0].train.differential_guidance_scale === null
                      ) {
                        // set default differential guidance scale to 3.0
                        setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                      }
                    }}
                  />
                  {jobConfig.config.process[0].train.differential_guidance_scale && (
                    <>
                      <NumberInput
                        label="Differential Guidance Scale"
                        className="pt-2"
                        value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                        placeholder="eg. 3.0"
                        min={0}
                      />
                    </>
                  )}
                </div>
                <div>
                {!isLlmModel && (
                  <FormGroup label="Other" className="pt-2">
                    <>
                      <Checkbox
                        label="Contrastive Guidance Loss"
                        docKey={'train.do_guidance_loss'}
                        className="pt-1"
                        checked={jobConfig.config.process[0].train.do_guidance_loss || false}
                        onChange={value => {
                          if (value) {
                            setJobConfig(true, 'config.process[0].train.do_guidance_loss');
                            if (!jobConfig.config.process[0].train.guidance_loss_target) {
                              setJobConfig(4.0, 'config.process[0].train.guidance_loss_target');
                            }
                          } else {
                            setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
                            setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
                          }
                        }}
                      />
                      {jobConfig.config.process[0].train.do_guidance_loss && (
                        <>
                          <NumberInput
                            label="Guidance Loss Target"
                            docKey={'train.guidance_loss_target'}
                            value={(jobConfig.config.process[0].train.guidance_loss_target as number) || 4.0}
                            onChange={value => setJobConfig(value, 'config.process[0].train.guidance_loss_target')}
                            placeholder="eg. 3.0"
                            min={0}
                          />
                        </>
                      )}
                    </>
                  </FormGroup>
                )}
                </div>
              </div>
            </Card>
          )}
        </div>
        <div>
          <Card title="Datasets">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                  <div className="absolute top-2 right-2 flex gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        const duplicated = objectCopy(dataset);
                        const datasets = [...jobConfig.config.process[0].datasets];
                        datasets.splice(i + 1, 0, duplicated);
                        setJobConfig(datasets, 'config.process[0].datasets');
                      }}
                      className="bg-gray-700 hover:bg-gray-600 rounded-full p-2 text-sm transition-colors"
                      title="Duplicate Dataset"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                          'config.process[0].datasets',
                        )
                      }
                      className="bg-red-600 hover:bg-red-700 text-white rounded-full p-2 text-sm transition-colors"
                      title="Remove Dataset"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <h2 className="text-lg font-bold mb-4">Dataset {i + 1}</h2>
                  <div className={datasetStyleClass}>
                    <div>
                      <SelectInput
                        label="Target Dataset"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={datasetOptions}
                      />
                      {modelArch?.additionalSections?.includes('datasets.control_path') && (
                        <SelectInput
                          label="Control Dataset"
                          docKey="datasets.control_path"
                          value={dataset.control_path ?? ''}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                          }
                          options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                        />
                      )}
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <>
                          <SelectInput
                            label="Control Dataset 1"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_1 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_1`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 2"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_2 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_2`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 3"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_3 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_3`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                        </>
                      )}
                      <NumberInput
                        label="LoRA Weight"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                      {(() => {
                        const selectedRes = dataset.resolution ?? [];
                        const repeatsRaw = dataset.num_repeats;
                        const scalarFallback = (typeof repeatsRaw === 'number' ? repeatsRaw : 1);
                        if (selectedRes.length <= 1) {
                          const scalarValue = Array.isArray(repeatsRaw)
                            ? (repeatsRaw[0] ?? 1)
                            : (repeatsRaw ?? 1);
                          return (
                            <NumberInput
                              label="Num Repeats"
                              value={scalarValue}
                              className="pt-2"
                              onChange={value =>
                                setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)
                              }
                              placeholder="eg. 1"
                              docKey={'dataset.num_repeats'}
                            />
                          );
                        }
                        const repeatsArr = Array.isArray(repeatsRaw)
                          ? selectedRes.map((_, idx) => repeatsRaw[idx] ?? scalarFallback)
                          : selectedRes.map(() => scalarFallback);
                        const commit = (next: number[]) => {
                          const allEqual = next.every(v => v === next[0]);
                          const value = allEqual ? (next[0] ?? 1) : next;
                          setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`);
                        };
                        return (
                          <FormGroup
                            label="Num Repeats (per resolution)"
                            className="pt-2"
                            docKey={'dataset.num_repeats'}
                          >
                            <div className="grid grid-cols-2 gap-2">
                              {selectedRes.map((res, idx) => (
                                <NumberInput
                                  key={res}
                                  label={`@ ${res}`}
                                  value={repeatsArr[idx] ?? 1}
                                  onChange={value => {
                                    const next = [...repeatsArr];
                                    next[idx] = (value as number) ?? 1;
                                    commit(next);
                                  }}
                                  placeholder="eg. 1"
                                  min={0}
                                />
                              ))}
                            </div>
                          </FormGroup>
                        );
                      })()}
                      <NumberInput
                        label="Batch Size"
                        value={dataset.batch_size ?? null}
                        className="pt-2"
                        onChange={value =>
                          setJobConfig(value == null ? undefined : value, `config.process[0].datasets[${i}].batch_size`)
                        }
                        placeholder={`${jobConfig.config.process[0].train.batch_size}`}
                        min={1}
                        allowEmpty
                      />
                    </div>
                    <div>
                      <TextInput
                        label="Default Caption"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <NumberInput
                        label="Caption Dropout Rate"
                        className="pt-2"
                        docKey="datasets.caption_dropout_rate"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                      <CreatableSelectInput
                        label="Caption Extension"
                        className="pt-2"
                        value={dataset.caption_ext || 'txt'}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_ext`)}
                        options={[
                          { value: 'txt', label: 'txt' },
                          { value: 'json', label: 'json' },
                          { value: 'caption', label: 'caption' },
                        ]}
                      />

                      {modelArch?.additionalSections?.includes('datasets.num_frames') && !dataset.auto_frame_count && (
                        <NumberInput
                          label="Num Frames"
                          className="pt-2"
                          docKey="datasets.num_frames"
                          value={dataset.num_frames}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                          placeholder="eg. 41"
                          min={1}
                          required
                        />
                      )}
                    </div>
                    <div>
                      <FormGroup label="Settings" className="">
                        <Checkbox
                          label="Cache Latents"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                          }
                        />
                        <Checkbox
                          label="Is Regularization"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                        {(modelArch?.additionalSections?.includes('datasets.control_path') ||
                          modelArch?.additionalSections?.includes('datasets.multi_control_paths')) && (
                          <Checkbox
                            label="Depth as Control"
                            checked={dataset.depth_as_control || false}
                            onChange={value =>
                              setJobConfig(value, `config.process[0].datasets[${i}].depth_as_control`)
                            }
                            docKey="datasets.depth_as_control"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.auto_frame_count') && (
                          <Checkbox
                            label="Auto Frame Count"
                            checked={dataset.auto_frame_count || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].auto_frame_count`)}
                            docKey="datasets.auto_frame_count"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                          <Checkbox
                            label="Do I2V"
                            checked={dataset.do_i2v || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                            docKey="datasets.do_i2v"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_audio') && (
                          <Checkbox
                            label="Do Audio"
                            checked={dataset.do_audio || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].do_audio`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].do_audio`);
                              }
                            }}
                            docKey="datasets.do_audio"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_normalize') && (
                          <Checkbox
                            label="Audio Normalize"
                            checked={dataset.audio_normalize || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_normalize`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_normalize`);
                              }
                            }}
                            docKey="datasets.audio_normalize"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_preserve_pitch') && (
                          <Checkbox
                            label="Audio Preserve Pitch"
                            checked={dataset.audio_preserve_pitch || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              }
                            }}
                            docKey="datasets.audio_preserve_pitch"
                          />
                        )}
                      </FormGroup>
                      {!isAudioModel && !isLlmModel && (
                      <FormGroup label="Flipping" docKey={'datasets.flip'} className="mt-2">
                        <Checkbox
                          label={
                            <>
                              Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_x || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                        />
                        <Checkbox
                          label={
                            <>
                              Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_y || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                        />
                      </FormGroup>
                      )}
                    </div>
                    {!isAudioModel && (
                    <div>
                      <FormGroup label="Resolutions" className="pt-2">
                        <div className="grid grid-cols-2 gap-2">
                          {[
                            [128, 256, 384, 512, 768],
                            [1024, 1280, 1328, 1536, 2048],
                          ].map(resGroup => (
                            <div key={resGroup[0]} className="space-y-2">
                              {resGroup.map(res => (
                                <Checkbox
                                  key={res}
                                  label={res.toString()}
                                  checked={dataset.resolution.includes(res)}
                                  onChange={value => {
                                    const cur = dataset.resolution;
                                    const removing = cur.includes(res);
                                    const resolutions = removing
                                      ? cur.filter(r => r !== res)
                                      : [...cur, res];
                                    setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                    if (Array.isArray(dataset.num_repeats)) {
                                      let nextRepeats: number[];
                                      if (removing) {
                                        const removeIdx = cur.indexOf(res);
                                        nextRepeats = dataset.num_repeats.filter((_, idx) => idx !== removeIdx);
                                      } else {
                                        const last = dataset.num_repeats[dataset.num_repeats.length - 1] ?? 1;
                                        nextRepeats = [...dataset.num_repeats, last];
                                      }
                                      const allEqual = nextRepeats.length > 0 && nextRepeats.every(v => v === nextRepeats[0]);
                                      const valueOut: number | number[] = nextRepeats.length === 0
                                        ? 1
                                        : (allEqual ? nextRepeats[0] : nextRepeats);
                                      setJobConfig(valueOut, `config.process[0].datasets[${i}].num_repeats`);
                                    }
                                  }}
                                />
                              ))}
                            </div>
                          ))}
                        </div>
                      </FormGroup>
                    </div>
                    )}
                  </div>
                  <details className="mt-3">
                      <summary className="cursor-pointer text-sm text-gray-400 hover:text-gray-200">
                        Per-Dataset Perceptual Anchoring Overrides
                      </summary>
                      <div className="mt-2 space-y-3 p-3 rounded bg-gray-700/50">
                        {/* Diffusion */}
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Diffusion</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.diffusion_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].diffusion_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.diffusion_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].diffusion_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.diffusion_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].diffusion_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Face Suppression" value={dataset.face_suppression_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].face_suppression_weight`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Supp. Expand" value={dataset.face_suppression_expand ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].face_suppression_expand`)} unsetOnEmpty placeholder="inherit" min={1.0} max={3.0} />
                          </div>
                        </div>
                        {/* Latent Perceptual — experimental, hidden for now
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Latent Perceptual</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.latent_perceptual_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].latent_perceptual_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.latent_perceptual_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].latent_perceptual_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.latent_perceptual_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].latent_perceptual_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        */}
                        {/* Identity */}
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Identity (ArcFace)</div>
                          <div className="grid grid-cols-4 gap-2">
                            <NumberInput label="Weight" value={dataset.identity_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].identity_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.identity_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].identity_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.identity_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].identity_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Min Cos" value={dataset.identity_loss_min_cos ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].identity_loss_min_cos`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        {/* Landmark — experimental, hidden for now
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Landmark</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.landmark_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].landmark_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                          </div>
                        </div>
                        */}
                        {/* Body Proportion */}
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Body Proportion</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.body_proportion_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_proportion_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.body_proportion_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_proportion_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.body_proportion_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_proportion_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        {/* Depth Consistency */}
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Depth Consistency</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.depth_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].depth_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.depth_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].depth_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.depth_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].depth_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        {/* Loss Alternation: alternate diffusion / depth per optimizer step.
                            Three states match the global Loss Split:
                              'auto' = key omitted (or null): inherit from global.
                              'diffusion_depth' = force on for this dataset.
                              'sum' = force off for this dataset (sum every step). */}
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Loss Alternation</div>
                          <SelectInput
                            label="Loss Split"
                            value={
                              dataset.loss_split === undefined || dataset.loss_split === null
                                ? 'auto'
                                : dataset.loss_split
                            }
                            onChange={value => {
                              if (value === 'auto') {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].loss_split`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].loss_split`);
                              }
                            }}
                            options={[
                              { value: 'auto', label: 'Auto (use global)' },
                              { value: 'diffusion_depth', label: 'Force on (Diffusion / Depth alternating)' },
                              { value: 'sum', label: 'Force off (sum every step)' },
                            ]}
                          />
                        </div>
                        {/* Subject Mask Region Weights */}
                        {jobConfig.config.process[0].subject_mask?.enabled && (
                          <div>
                            <div className="text-xs font-medium text-gray-400 mb-1">Subject Mask Regions</div>
                            <div className="grid grid-cols-4 gap-2">
                              <NumberInput label="Background" value={dataset.background_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].background_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                              <NumberInput label="Clothing" value={dataset.clothing_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].clothing_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                              <NumberInput label="Body" value={dataset.body_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                              <Checkbox label="Restrict Perc." checked={dataset.perceptual_restrict_to_body ?? false} onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].perceptual_restrict_to_body`)} />
                            </div>
                          </div>
                        )}
                        {/* Body Shape, Normal Map, VAE Anchor — experimental, hidden for now
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Body Shape</div>
                          <div className="grid grid-cols-4 gap-2">
                            <NumberInput label="Weight" value={dataset.body_shape_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_shape_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.body_shape_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_shape_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.body_shape_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_shape_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Min Cos" value={dataset.body_shape_loss_min_cos ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].body_shape_loss_min_cos`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">Normal Map</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.normal_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].normal_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.normal_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].normal_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.normal_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].normal_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-gray-400 mb-1">VAE Anchor</div>
                          <div className="grid grid-cols-3 gap-2">
                            <NumberInput label="Weight" value={dataset.vae_anchor_loss_weight ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].vae_anchor_loss_weight`)} unsetOnEmpty placeholder="inherit" min={0} />
                            <NumberInput label="Min t" value={dataset.vae_anchor_loss_min_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].vae_anchor_loss_min_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                            <NumberInput label="Max t" value={dataset.vae_anchor_loss_max_t ?? null} onChange={value => setJobConfig(value === null || value === undefined ? undefined : value, `config.process[0].datasets[${i}].vae_anchor_loss_max_t`)} unsetOnEmpty placeholder="inherit" min={0} max={1} />
                          </div>
                        </div>
                        */}
                      </div>
                  </details>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const newDataset = objectCopy(defaultDatasetConfig);
                  // automaticallt add the controls for a new dataset
                  const controls = modelArch?.controls ?? [];
                  newDataset.controls = controls;
                  // arch dataset defaults (datasets[x].*) apply to added datasets too, not just at arch switch
                  for (const key in modelArch?.defaults ?? {}) {
                    const marker = 'datasets[x].';
                    const idx = key.indexOf(marker);
                    if (idx !== -1) {
                      (newDataset as any)[key.slice(idx + marker.length)] = modelArch!.defaults![key][0];
                    }
                  }
                  setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                }}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Add Dataset
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="Sample">
            <div className={sampleTopStyleClass}>
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <NumberInput
                  label="Sample Start Step"
                  value={jobConfig.config.process[0].sample.sample_start_step ?? 0}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_start_step')}
                  placeholder="eg. 0"
                  className="pt-2"
                  min={0}
                  required
                />
                {!isLlmModel && (
                  <>
                    <SelectInput
                      label="Sampler"
                      className="pt-2"
                      value={jobConfig.config.process[0].sample.sampler}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                      options={[
                        { value: 'flowmatch', label: 'FlowMatch' },
                        { value: 'ddpm', label: 'DDPM' },
                      ]}
                    />
                    <NumberInput
                      label="Guidance Scale"
                      value={jobConfig.config.process[0].sample.guidance_scale}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                      placeholder="eg. 1.0"
                      className="pt-2"
                      min={0}
                      required
                    />
                    <NumberInput
                      label="Sample Steps"
                      value={jobConfig.config.process[0].sample.sample_steps}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                      placeholder="eg. 1"
                      className="pt-2"
                      min={1}
                      required
                    />
                  </>
                )}
              </div>

              {!isAudioModel && !isLlmModel && (
                <div>
                  <NumberInput
                    label="Width"
                    value={jobConfig.config.process[0].sample.width}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                    placeholder="eg. 1024"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Height"
                    value={jobConfig.config.process[0].sample.height}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                    placeholder="eg. 1024"
                    className="pt-2"
                    min={0}
                    required
                  />
                  {isVideoModel && (
                    <div>
                      <NumberInput
                        label="Num Frames"
                        value={jobConfig.config.process[0].sample.num_frames}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                      <NumberInput
                        label="FPS"
                        value={jobConfig.config.process[0].sample.fps}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                    </div>
                  )}
                </div>
              )}

              {!isLlmModel && (
                <div>
                  <NumberInput
                    label="Seed"
                    value={jobConfig.config.process[0].sample.seed}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                    placeholder="eg. 0"
                    min={0}
                    required
                  />
                  {modelArch?.additionalSections?.includes('sample.duration') && (
                    <NumberInput
                      label="Duration (seconds)"
                      value={jobConfig.config.process[0].sample.duration ?? 120}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.duration')}
                      placeholder="eg. 120"
                      className="pt-2"
                      min={1}
                      required
                    />
                  )}
                  <Checkbox
                    label="Walk Seed"
                    className="pt-4 pl-2"
                    checked={jobConfig.config.process[0].sample.walk_seed}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                  />
                </div>
              )}
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            <div className="pt-2 mb-2 flex items-center justify-between">
              <label className="block text-xs text-gray-300">
                Sample Prompts ({jobConfig.config.process[0].sample.samples.length})
              </label>
              {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                <button
                  type="button"
                  disabled={jobConfig.config.process[0].sample.samples.length === 0}
                  onClick={() => {
                    const sampleCfg = jobConfig.config.process[0].sample;
                    const items = sampleCfg.samples
                      .map((s, i) => ({
                        index: i,
                        prompt: s.prompt || '',
                        aspectRatio: toAspectRatio(s.width || sampleCfg.width, s.height || sampleCfg.height),
                      }))
                      .filter(it => it.prompt.trim() !== '');
                    if (items.length === 0) return;
                    openUpsamplePromptsModal(items, (index, newPrompt) => {
                      setJobConfig(newPrompt, `config.process[0].sample.samples[${index}].prompt`);
                    });
                  }}
                  className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md inline-flex items-center gap-2"
                >
                  <Wand2 className="w-4 h-4" />
                  Upsample Prompts
                </button>
              )}
            </div>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        {modelArch?.sampleTags && taggedSampleArr && modelArchTagSections ? (
                          <>
                            {modelArchTagSections.map((sampleTagSection, sti) => (
                              <div key={sti} className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                                {Object.entries(sampleTagSection).map(([tagKey, tag]) => (
                                  <div key={tagKey} className="mb-2">
                                    {tag.type === 'text' && (
                                      <TextInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'multiline' && (
                                      <TextAreaInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'number' && (
                                      <NumberInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </>
                        ) : (
                          <>
                            {modelArch?.hasMultiLinePrompts ? (
                              <TextAreaInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            ) : (
                              <TextInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder={isLlmModel ? 'blank = LLM Prompt' : 'Enter prompt'}
                                required={!isLlmModel}
                              />
                            )}
                          </>
                        )}

                        {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                          <div className="mt-2">
                            <button
                              type="button"
                              onClick={() => {
                                const sampleCfg = jobConfig.config.process[0].sample;
                                openPromptBoxEditor({
                                  prompt: sample.prompt || '',
                                  aspectRatio: toAspectRatio(
                                    sample.width || sampleCfg.width,
                                    sample.height || sampleCfg.height,
                                  ),
                                  title: `Prompt #${i + 1}`,
                                  onApply: newPrompt =>
                                    setJobConfig(newPrompt, `config.process[0].sample.samples[${i}].prompt`),
                                });
                              }}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-gray-600 text-gray-300 hover:bg-gray-800 transition-colors"
                            >
                              <SquareDashed className="w-3.5 h-3.5" />
                              Edit caption &amp; boxes
                            </button>
                          </div>
                        )}

                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          {!isAudioModel && !isLlmModel && (
                            <TextInput
                              label={`Width`}
                              value={sample.width ? `${sample.width}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].width;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                  } else {
                                    console.warn('Invalid width value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                            />
                          )}
                          {!isAudioModel && !isLlmModel && (
                            <TextInput
                              label={`Height`}
                              value={sample.height ? `${sample.height}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].height;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                  } else {
                                    console.warn('Invalid height value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                            />
                          )}
                          {!isLlmModel && (
                            <TextInput
                              label={`Seed`}
                              value={sample.seed ? `${sample.seed}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].seed;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                  } else {
                                    console.warn('Invalid seed value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                            />
                          )}
                          <TextInput
                            label={`LoRA Scale`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />
                        </div>
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="Control Images" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'].map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={`Add Control Image ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          instruction={isLlmModel ? 'Add Media' : undefined}
                          allowAudio={isLlmModel}
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              Add Prompt
            </button>
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <AddSingleImageModal />
    </>
  );
}
