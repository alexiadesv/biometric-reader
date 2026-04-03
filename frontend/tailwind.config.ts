import type { Config } from "tailwindcss";

/** Accent palette; page chrome uses background + surface below */
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#EBECE1",
        surface: "#F7F7EE",
        /** Near-white inner cards (contrast vs tinted highlights) */
        panel: "#FEFEFB",
        ink: "#2A2422",
        "ink-muted": "#4D4542",
        "ink-faint": "#6B625E",
        line: "#D4D9E4",
        obv: {
          cream: "#F7F7EE",
          rose: "#F0B4C4",
          yellow: "#FDE047",
          sage: "#86C4A8",
          peach: "#FCA5A5",
          lavender: "#C4B5FD",
          mint: "#86C4A8",
          accent: "#38BDF8"
        }
      },
      boxShadow: {
        soft: "0 24px 60px rgba(15, 23, 42, 0.08), 0 10px 28px rgba(56, 189, 248, 0.06)",
        card: "0 1px 3px rgba(15, 23, 42, 0.05), 0 6px 20px rgba(99, 102, 241, 0.06)",
        glow: "0 8px 32px rgba(56, 189, 248, 0.25)"
      }
    }
  },
  plugins: []
};

export default config;
