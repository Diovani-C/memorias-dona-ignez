import { readdir } from "node:fs/promises";
import { join, parse } from "node:path";

// --- CONFIGURATION ---
// Adjust these paths to match your project structure
const IMAGES_ROOT_DIR = "./images";
const FACES_INFO_FILE = "./face-data.json";
const ALLOWED_FACES_FILE = "./images-filters.json";
const OUTPUT_FILE = "./images-metadata.json";
// --- END CONFIGURATION ---

// Define TypeScript types for our data structures
interface FaceInfo {
  img: string;
  faces: string[];
}

interface AllowedFace {
  id: string;
  displayValue: string;
}

interface OutputItem {
  slug: string;
  src: string;
  alt: string;
  raw?: string; // This property is optional
  filters: string[];
}

/**
 * Main function to generate the image metadata JSON.
 */
async function generateImageMetadata() {
  console.log("🚀 Starting image metadata generation...");

  // 1. Load and parse the helper JSON files.
  const facesInfo: FaceInfo[] = await Bun.file(FACES_INFO_FILE).json();
  const allowedFacesData: { faces: AllowedFace[] } =
    await Bun.file(ALLOWED_FACES_FILE).json();

  // 2. Create efficient data structures for quick lookups.
  // A Set for O(1) checking of allowed faces.
  const allowedFacesSet = new Set(
    allowedFacesData.faces.map((face) => face.id),
  );
  // A Map for O(1) lookup of faces by image filename.
  const facesMap = new Map<string, string[]>(
    facesInfo.map((info) => [info.img, info.faces]),
  );

  console.log(`✅ Loaded ${allowedFacesSet.size} allowed faces.`);
  console.log(`✅ Loaded face info for ${facesMap.size} images.`);

  // 3. Scan the images directory and process each category.
  const finalOutput: OutputItem[] = [];
  const categories = await readdir(IMAGES_ROOT_DIR, { withFileTypes: true });

  categories.sort((file1, file2) => file1.name.localeCompare(file2.name));

  for (const category of categories) {
    // We only process subdirectories.
    if (!category.isDirectory()) continue;

    const categoryName = category.name;
    const categoryPath = join(IMAGES_ROOT_DIR, categoryName);
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

      // D. Check for a corresponding "raw" image.
      const rawPath = join(categoryPath, "raw", imageFilename);
      const raw = (await Bun.file(rawPath).exists())
        ? `./${categoryName}/raw/${imageFilename}`
        : undefined;

      // E. Build the filters array.
      const filters: string[] = [categoryName];
      const imageFaces = facesMap.get(imageFilename) ?? [];

      // Use a Set to handle duplicates from the source JSON, then filter.
      const uniqueValidFaces = [...new Set(imageFaces)].filter((face) =>
        allowedFacesSet.has(face),
      );

      filters.push(...uniqueValidFaces);

      // F. Assemble the final object for this image.
      const outputItem: OutputItem = {
        slug: baseName,
        src,
        alt: alt.trim(),
        filters,
      };

      // Only add the 'raw' key if a raw image was found.
      if (raw) {
        outputItem.raw = raw;
      }

      finalOutput.push(outputItem);
      console.log(`   ✔️ Processed ${imageFilename}`);
    }
  }

  // 4. Write the final array to the output JSON file.
  await Bun.write(OUTPUT_FILE, JSON.stringify(finalOutput, null, 2));
  console.log(
    `\n🎉 Success! Generated ${OUTPUT_FILE} with ${finalOutput.length} entries.`,
  );
}

// Run the script and catch any potential errors.
generateImageMetadata().catch((error) => {
  console.error("❌ An error occurred:", error);
});
