// 1. Import utilities from `astro:content`
import { defineCollection, z } from "astro:content";

// 2. Import loader(s)
import { file } from "astro/loaders";

// 3. Define your collection(s)
const images = defineCollection({
  loader: file("src/data/images-metadata.json"),
  schema: ({ image }) =>
    z.object({
      slug: z.string(),
      src: image(),
      alt: z.string(),
      raw: z.string().optional(),
      filters: z.array(z.string()),
    }),
});

const faces = defineCollection({
  loader: file("src/data/images-filters.json", {
    parser: (text) => JSON.parse(text).faces,
  }),
  schema: z.object({
    id: z.string(),
    displayValue: z.string(),
  }),
});

const categories = defineCollection({
  loader: file("src/data/images-filters.json", {
    parser: (text) => JSON.parse(text).categories,
  }),
  schema: z.object({
    id: z.string(),
    displayValue: z.string(),
  }),
});

// 4. Export a single `collections` object to register your collection(s)
export const collections = {
  images,
  faces,
  categories,
};
