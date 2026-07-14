import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Python API runs on :8000; proxy /api during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: { outDir: 'dist' },
})
