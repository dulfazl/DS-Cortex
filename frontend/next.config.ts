import type { NextConfig } from "next";

// The dashboard is client-driven (live SSE, approvals), so it uses the previous caching model
// rather than Cache Components. `standalone` output keeps the Docker image small.
const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
