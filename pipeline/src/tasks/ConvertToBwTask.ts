/**
 * @file Task to convert images to black and white (grayscale) using the Sharp library.
 */

import sharp from "sharp";
import { join, parse } from "path";
import type { TaskFunction, FileDataType } from "../types.js";
import { getInputFiles } from "./utils.js";

/**
 * The core conversion function using Sharp.
 * @param {FileDataType} file - The file to process.
 * @param {string} outputPath - The directory to save the converted image.
 */
async function convertToGrayscale(
  file: FileDataType,
  outputPath: string,
): Promise<void> {
  try {
    const originalName = parse(file.name).name;
    const outputFilePath = join(outputPath, `${originalName}.png`);

    console.log(`   -> Converting ${file.name} to black and white...`);

    await sharp(file.data).grayscale().toFile(outputFilePath);
  } catch (error) {
    console.error(`Failed to convert ${file.name}. Skipping.`, error);
  }
}

/**
 * Creates a task function for converting images to black and white.
 * @returns {TaskFunction} The function to be executed by the CLI.
 */
export function createConvertToBwTask(): TaskFunction {
  return async ({ inputPath, outputPath }) => {
    const inputFiles = await getInputFiles(inputPath);

    if (inputFiles.length === 0) {
      throw new Error("No input files found to process.");
    }

    console.log(
      `Found ${inputFiles.length} file(s) to convert to black and white.`,
    );

    inputFiles.sort((file1, file2) => file1.name.localeCompare(file2.name));

    // Create an array of conversion promises to run them concurrently.
    const conversionPromises = inputFiles.map((file) =>
      convertToGrayscale(file, outputPath),
    );

    // Wait for all images to be processed.
    await Promise.all(conversionPromises);
  };
}
