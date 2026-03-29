import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#020617",
        surface: "#020617",
        accent: "#38bdf8"
      },
      boxShadow: {
        soft: "0 18px 45px rgba(15,23,42,0.65)"
      }
    }
  },
  plugins: []
};

export default config;
