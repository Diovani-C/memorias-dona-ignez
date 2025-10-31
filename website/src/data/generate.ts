import { readdir } from "node:fs/promises";
import { join, parse } from "node:path";

const IMAGES_ROOT_DIR = "./images";
const FACES_INFO_FILE = "face-data.json";
const OUTPUT_FILE = "./images-metadata.json";

interface FaceInfo {
  img: string;
  faces: string[];
}

interface OutputItem {
  slug: string;
  src: string;
  alt: string;
  raw?: string;
  filters: string[];
}

async function generateImageMetadata() {
  console.log("🚀 Starting image metadata generation...");

  const finalOutput: OutputItem[] = [];
  const categories = await readdir(IMAGES_ROOT_DIR, { withFileTypes: true });

  categories.sort((file1, file2) => file1.name.localeCompare(file2.name));

  for (const category of categories) {
    if (!category.isDirectory()) continue;

    const categoryName = category.name;
    const categoryPath = join(IMAGES_ROOT_DIR, categoryName);

    const facesInfo: FaceInfo[] = await Bun.file(
      join(categoryPath, FACES_INFO_FILE),
    ).json();

    const facesMap = new Map<string, string[]>(
      facesInfo.map((info) => [info.img, info.faces]),
    );

    console.log(`\n🔍 Processing category: ${categoryName}`);

    const files = await readdir(categoryPath);
    const imageFiles = files.filter((file) =>
      /\.(jpe?g|png|webp)$/i.test(file),
    );

    imageFiles.sort();

    for (const imageFilename of imageFiles) {
      const { name: baseName } = parse(imageFilename);

      const src = `./images/${categoryName}/${imageFilename}`;

      const altTextPath = join(categoryPath, `${baseName}.txt`);
      const alt = (await Bun.file(altTextPath).exists())
        ? await Bun.file(altTextPath).text()
        : "";

      if (!alt) {
        console.warn(`   ⚠️ Alt text not found for ${imageFilename}`);
      }

      const rawPath = join(categoryPath, "raw", imageFilename);
      const raw = (await Bun.file(rawPath).exists())
        ? `./images/${categoryName}/raw/${imageFilename}`
        : undefined;

      const filters: string[] = [categoryName];
      const imageFaces = facesMap.get(imageFilename) ?? [];

      filters.push(...imageFaces);

      const outputItem: OutputItem = {
        slug: baseName,
        src,
        alt: alt.trim(),
        filters,
      };

      if (raw) {
        outputItem.raw = raw;
      }

      finalOutput.push(outputItem);
      console.log(`   ✔️ Processed ${imageFilename}`);
    }
  }

  await Bun.write(OUTPUT_FILE, JSON.stringify(finalOutput, null, 2));
  console.log(
    `\n🎉 Success! Generated ${OUTPUT_FILE} with ${finalOutput.length} entries.`,
  );
}

generateImageMetadata().catch((error) => {
  console.error("❌ An error occurred:", error);
});
