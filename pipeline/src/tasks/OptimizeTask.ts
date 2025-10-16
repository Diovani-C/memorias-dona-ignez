/**
 * @file Task to optimize images for web usage using the Sharp library.
 * It resizes, removes metadata, and converts images to the WebP format.
 */

import sharp from "sharp";
import { join, parse } from "path";
import type { TaskFunction, FileDataType } from "../types.js";
import { getInputFiles } from "./utils.ts";

// --- Configuration ---

// The maximum width or height for the output image.
// Images larger than this will be shrunk, while smaller images will not be enlarged.
const MAX_DIMENSION = 3840;

// The quality setting for WebP (1-100). 80 is a great balance.
const WEBP_QUALITY = 90;

/**
 * The core optimization function using Sharp.
 * @param {FileDataType} file - The file to process.
 * @param {string} outputPath - The directory to save the optimized image.
 */
async function optimizeImage(
  file: FileDataType,
  outputPath: string,
): Promise<void> {
  try {
    const originalName = parse(file.name).name;
    const outputFilePath = join(outputPath, `${originalName}.webp`);

    console.log(`   -> Optimizing ${file.name}...`);

    await sharp(file.data)
      // Auto-rotates the image based on EXIF data, then strips all metadata.
      .rotate()
      // Add a very light blur to reduce image details and size
      .blur(0.3)
      // Resize the image to fit within the max dimensions.
      .resize({
        width: MAX_DIMENSION,
        height: MAX_DIMENSION,
        fit: "inside", // Ensures the aspect ratio is maintained.
        withoutEnlargement: true, // Prevents upscaling of smaller images.
      })
      // Convert to WebP with specific quality settings.
      .webp({
        quality: WEBP_QUALITY,
        effort: 6,
      })
      // Save the processed image to the output file.
      .toFile(outputFilePath);
  } catch (error) {
    console.error(`Failed to optimize ${file.name}. Skipping.`, error);
  }
}

/**
 * Creates a task function for optimizing images for the web.
 * @returns {TaskFunction} The function to be executed by the CLI.
 */
export function createOptimizeImageTask(): TaskFunction {
  return async ({ inputPath, outputPath }) => {
    // This assumes `getInputFiles` exists and is imported, as in your `cli.ts`.
    // We'll need to add it to the CLI to make it available here.
    const inputFiles = await getInputFiles(inputPath);

    if (inputFiles.length === 0) {
      throw new Error("No input files found to process.");
    }

    console.log(`Found ${inputFiles.length} file(s) to optimize.`);

    // Create an array of optimization promises to run them concurrently.
    const optimizationPromises = inputFiles.map((file) =>
      optimizeImage(file, outputPath),
    );

    // Wait for all images to be processed.
    await Promise.all(optimizationPromises);
  };
}
