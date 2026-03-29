/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: "jsdom",
  moduleFileExtensions: ["ts", "tsx", "js", "jsx"],
  transform: {
    "^.+\\.(ts|tsx)$": [
      "ts-jest",
      {
        tsconfig: "tsconfig.json",
        jsx: "react-jsx"
      }
    ]
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  testMatch: ["**/__tests__/**/*.test.(ts|tsx)"],
  testPathIgnorePatterns: ["/node_modules/", "/.next/"]
};

