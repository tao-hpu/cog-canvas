/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['react-force-graph-2d'],
  // Same root the previous experimental.turbo setting declared.
  // `next build` still uses webpack unless --turbopack is passed.
  turbopack: {
    root: process.cwd(),
  },
}

module.exports = nextConfig
