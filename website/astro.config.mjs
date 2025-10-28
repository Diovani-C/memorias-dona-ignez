// @ts-check
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";

// https://astro.build/config
export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },

  prefetch: true,
  integrations: [
    sitemap({
      filter: (page) => page !== "https://memoriasdonaignez.com/face/",
    }),
  ],
  site: "https://memoriasdonaignez.com",
});

