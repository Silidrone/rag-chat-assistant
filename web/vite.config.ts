import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Flask service runs on 5000 (bare) or 5001 (docker compose). Proxying
// keeps the browser on one origin, so the API needs no CORS handling.
const API = process.env.API_URL ?? "http://localhost:5000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3041,
    proxy: {
      "/ask": { target: API, changeOrigin: true },
      "/health": { target: API, changeOrigin: true },
      "/articles": { target: API, changeOrigin: true },
    },
  },
});
