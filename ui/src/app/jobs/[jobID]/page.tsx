'use client';

import { useMemo, useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { redirect } from 'next/navigation';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import JobMetricsGraph from '@/components/JobMetricsGraph';
import JobMetricsCompareGraph from '@/components/JobMetricsCompareGraph';
import DepthPreviews from '@/components/DepthPreviews';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';

type PageKey = 'overview' | 'samples' | 'depth_previews' | 'config' | 'loss_log' | 'metrics_new' | 'metrics_compare';

interface Page {
  name: string;
  value: PageKey;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
  /** Hide the tab unless the predicate (run against the loaded job) returns true. */
  condition?: (job: Job) => boolean;
}

function hasDepthPreviews(job: Job): boolean {
  if (!job.job_config) return false;
  try {
    const cfg = JSON.parse(job.job_config) as JobConfig;
    return (cfg.config?.process?.[0]?.depth_consistency?.preview_every ?? 0) > 0;
  } catch {
    return false;
  }
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    component: JobOverview,
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    value: 'samples',
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
  },
  {
    name: 'Depth Previews',
    value: 'depth_previews',
    component: DepthPreviews,
    mainCss: 'pt-24',
    condition: hasDepthPreviews,
  },
  {
    name: 'Loss Graph',
    value: 'loss_log',
    component: JobLossGraph,
    mainCss: 'pt-24',
  },
  {
    // Step 5: parallel-installed alongside the legacy "Loss Graph". Reads
    // the canonical `subsystem/kind/variant` namespace + per-sample
    // breakdown payloads added in steps 3-4.
    name: 'Metrics (new)',
    value: 'metrics_new',
    component: JobMetricsGraph,
    mainCss: 'pt-24',
  },
  {
    // Cross-job comparison: pick a metric, fan it across N selected jobs.
    // Anchored on the current job; additional jobs picked from the multi-
    // select. Same fetch pipeline as Metrics (new), N-way parallel.
    name: 'Compare Jobs',
    value: 'metrics_compare',
    component: JobMetricsCompareGraph,
    mainCss: 'pt-24',
  },
  {
    name: 'Config File',
    value: 'config',
    component: JobConfigViewer,
    mainCss: 'pt-[80px] px-0 pb-0',
  },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);
  const [pageKey, setPageKey] = useState<PageKey>('overview');

  const visiblePages = useMemo(() => (job ? pages.filter(p => !p.condition || p.condition(job)) : pages.filter(p => !p.condition)), [job]);
  // If the previously selected tab no longer applies (e.g. preview_every was
  // turned off after the user landed on it), bounce back to overview.
  const page = visiblePages.find(p => p.value === pageKey) ?? visiblePages[0];
  const effectivePageKey = page?.value ?? 'overview';

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Job: {job?.name}</h1>
        </div>
        <div className="flex-1"></div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              redirect('/jobs');
            }}
            autoStartQueue={true}
          />
        )}
      </TopBar>
      <MainContent className={page?.mainCss}>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {job && (
          <>
            {visiblePages.map(p => {
              const Component = p.component;
              return p.value === effectivePageKey ? <Component key={p.value} job={job} /> : null;
            })}
          </>
        )}
      </MainContent>
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-2 text-sm">
        {visiblePages.map(p => (
          <Button
            key={p.value}
            onClick={() => setPageKey(p.value)}
            className={`px-4 py-1 h-8  ${p.value === effectivePageKey ? 'bg-gray-300 dark:bg-gray-700' : ''}`}
          >
            {p.name}
          </Button>
        ))}
        {page?.menuItem && (
          <>
            <div className="flex-grow"></div>
            <page.menuItem job={job} />
          </>
        )}
      </div>
    </>
  );
}
