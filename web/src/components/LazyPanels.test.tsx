import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';
import { DataSandboxSuspense, SettingsPanelSuspense } from './LazyPanels';

const container = document.createElement('div');
let root: ReturnType<typeof createRoot>;

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

it('does not cover the workspace with loading overlays for closed panels', () => {
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(<>
    <SettingsPanelSuspense open={false} onClose={() => {}} refreshModels={async () => {}}
      pickedModels={{ chat: undefined }} resetAll={() => {}} clearSandboxData={() => {}} />
    <DataSandboxSuspense open={false} onClose={() => {}} addFiles={async () => {}}
      loadSampleData={async () => {}} clearSandboxData={() => {}}
      removeSandboxDocument={() => {}} removeSandboxDataset={() => {}} />
  </>));
  expect(container.querySelector('[role="status"]')).toBeNull();
  expect(container.textContent).toBe('');
});
