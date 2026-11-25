import nextCoreWebVitals from "eslint-config-next/core-web-vitals";

const config = Array.isArray(nextCoreWebVitals)
  ? nextCoreWebVitals
  : nextCoreWebVitals.default;

const eslintConfig = [
  ...config,
  {
    ignores: [".next/**", "node_modules/**"],
  },
];

export default eslintConfig;
