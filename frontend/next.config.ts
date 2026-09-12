import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal self-contained server bundle for the Dockerfile below -- not
  // needed for a Vercel deploy, which builds natively, but keeps `docker
  // compose up` (this repo's "reproducible local run" story) working too.
  output: "standalone",
};

export default nextConfig;
