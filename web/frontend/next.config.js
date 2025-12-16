/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['react-force-graph-2d'],
  // Disable Turbopack for build to avoid path encoding issues
  experimental: {
    turbo: {
      root: process.cwd(),
    },
  },
}

module.exports = nextConfig
