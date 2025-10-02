import type { FileDataType } from "./types";
import Bun from "bun";
import { resolve, basename, join } from "path";
import { readdir, mkdir } from "node:fs/promises";
import { stat } from "fs/promises";

/**
 * Reads a single file and returns it in the standard FileDataType format.
 * @param {string} filePath - The full path to the file.
 * @returns {Promise<FileDataType>} A promise that resolves to the file data object.
 */
async function getFile(filePath: string): Promise<FileDataType> {
  try {
    const fileData = Bun.file(filePath);

    return {
      name: basename(filePath),
      data: await fileData.arrayBuffer(),
      mimeType: fileData.type,
    };
  } catch (error) {
    console.error("Error reading the file", error);
    throw error;
  }
}

/**
 * Takes an input path (which can be a single file or a directory) and
 * returns an array of all files found.
 * @param {string} inputPath - The path to a file or directory.
 * @returns {Promise<FileDataType[]>} A promise that resolves to an array of file data objects.
 */
export async function getInputFiles(
  inputPath: string,
): Promise<FileDataType[]> {
  const absoluteInputPath = resolve(inputPath);
  const stats = await stat(absoluteInputPath);

  // If the path is a single file, read it and return as a single-element array.
  if (stats.isFile()) {
    return [await getFile(absoluteInputPath)];
  }

  // If the path is a directory, read all files within it.
  if (stats.isDirectory()) {
    const fileNames = await readdir(absoluteInputPath);
    // Create a promise for each file reading operation.
    const filePromises = fileNames.map((fileName) => {
      const filePath = join(absoluteInputPath, fileName);
      return getFile(filePath);
    });

    // Wait for all file reading promises to resolve concurrently.
    return await Promise.all(filePromises);
  }

  // If the path is neither a file nor a directory, throw an error.
  throw new Error(
    `Input path is not a valid file or directory: ${absoluteInputPath}`,
  );
}

async function saveBinaryFile(name: string, content: Buffer) {
  Bun.write(name, content);
}

export async function saveBinaryFilesToFolder(
  folder: string,
  files: { name: string; content: Buffer }[],
) {
  try {
    const newFolder = await mkdir(folder, { recursive: true });
    if (newFolder == undefined) throw "Error creating the output folder";

    for (let { name, content } of files) {
      await saveBinaryFile(`${newFolder}/${name}`, content);
    }
  } catch (error) {
    console.error("Error saving Binary Files:", error);
    throw error;
  }
}
