// 1. Import utilities from `astro:content`
import { defineCollection, z } from "astro:content";

// 2. Import loader(s)
import { file } from "astro/loaders";

// 3. Define your collection(s)
const images = defineCollection({
  loader: file("src/data/images-metadata.json"),
  schema: z.object({
    slug: z.string(),
    src: z.string(),
    alt: z.string(),
    raw: z.string().optional(),
    categories: z.array(z.string()),
  }),
});

// 4. Export a single `collections` object to register your collection(s)
export const collections = { images };
