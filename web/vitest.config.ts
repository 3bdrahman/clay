import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    globals: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'dist'],
    testTimeout: 30000,
    coverage: {
      enabled: false,
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/lib/**', 'src/services/**'],
    },
  },
});
