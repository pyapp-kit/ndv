import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";

export default defineConfig({
  plugins: [anywidget()],
  build: {
    target: ['chrome112', 'edge112', 'firefox121', 'safari16.4'],
    outDir: "../src/ndv/views/_jupyter/static",
    emptyOutDir: true,
    lib: {
      entry: "src/index.js",
      formats: ["es"],
    },
    rollupOptions: {
      output: {
        // Must inline everything — anywidget loads ESM as a blob URL
        // which can't resolve relative chunk imports.
        inlineDynamicImports: true,
      },
    },
  },
});
