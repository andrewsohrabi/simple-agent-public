import { defineConfig, devices } from '@playwright/test'

const backendPort = process.env.PLAYWRIGHT_API_PORT ?? '8127'
const frontendPort = process.env.PLAYWRIGHT_FRONTEND_PORT ?? '3127'

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {
    timeout: 20_000,
  },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: [
    {
      command: 'uv run serve',
      cwd: '..',
      url: `http://127.0.0.1:${backendPort}/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        ...process.env,
        PORT: backendPort,
        HOST: '127.0.0.1',
        UVICORN_RELOAD: 'false',
        UV_CACHE_DIR: process.env.UV_CACHE_DIR ?? '/private/tmp/uv-cache',
        QMS_USE_HASH_EMBEDDINGS: 'true',
      },
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${frontendPort}`,
      url: `http://127.0.0.1:${frontendPort}`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        ...process.env,
        VITE_API_URL: `http://127.0.0.1:${backendPort}`,
      },
    },
  ],
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
