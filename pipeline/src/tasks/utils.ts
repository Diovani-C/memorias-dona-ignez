/**
 * @file Shared utility functions for interacting with the Google Gemini API.
 * This includes API client initialization, batch file processing, and job monitoring.
 */

import { GoogleGenAI, type Content } from "@google/genai";
import { resolve, join, basename } from "path";
import { stat, appendFile, writeFile } from "fs/promises";
import type { FileDataType } from "../types";
import Bun from "bun";
import { readdir } from "node:fs/promises";

// --- Shared Type Definitions ---

export interface RequestType {
  key: string;
  request: { contents: Content };
}

// --- API Client Initialization ---

/**
 * Initializes and returns the GoogleGenAI client.
 * It centralizes the API key check.
 * @returns {GoogleGenAI} The initialized GoogleGenAI instance.
 */
export function getGoogleAI(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY environment variable not found. Please set it before running the script.",
    );
  }
  return new GoogleGenAI({ apiKey });
}

// --- Batch Job Helpers ---

/**
 * Writes an array of request objects to a JSONL file in the specified output path.
 * @param {RequestType[]} requests - The request objects to write.
 * @param {string} outputPath - The directory to save the JSONL file in.
 * @returns {Promise<string>} The full path to the newly created JSONL file.
 */
export async function writeBatchRequestsToFile(
  requests: RequestType[],
  outputPath: string,
): Promise<string> {
  try {
    const stats = await stat(outputPath);
    if (!stats.isDirectory()) {
      throw new Error(`Output path is not a directory: ${outputPath}`);
    }

    const filePath = join(outputPath, `${crypto.randomUUID()}.jsonl`);
    const jsonlContent = requests.map((req) => JSON.stringify(req)).join("\n");

    await writeFile(filePath, jsonlContent);

    console.log(
      `Successfully wrote ${requests.length} batch requests to ${filePath}`,
    );
    return filePath;
  } catch (error) {
    throw new Error(
      `Error writing batch requests to file: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

/**
 * Saves essential batch job information to a central log file (batch_jobs.log).
 * @param {string} batchJobName - The unique name of the created batch job.
 * @param {string} jsonlFilePath - Path to the input JSONL file for this job.
 */
export async function saveBatchJobInfo(
  batchJobName: string,
  jsonlFilePath: string,
): Promise<void> {
  const logFilePath = resolve(process.cwd(), "batch_jobs.log");
  const logEntry = `${new Date().toISOString()} | JobName: ${batchJobName} | InputFile: ${jsonlFilePath}\n`;
  try {
    await appendFile(logFilePath, logEntry);
    console.log(`Batch job info saved to ${logFilePath}`);
  } catch (error) {
    // This is a non-critical function, so we log a warning instead of throwing an error.
    console.warn(`Warning: Could not write to batch log file. ${error}`);
  }
}

/**
 * Polls the Gemini API to monitor a batch job's status until it completes.
 * @param {GoogleGenAI} ai - The initialized GoogleGenAI client instance.
 * @param {string} batchJobName - The name of the batch job to monitor.
 * @returns The final batch job object once it has finished.
 */
export async function monitoringBatchJobStatus(
  ai: GoogleGenAI,
  batchJobName: string,
) {
  const completedStates = new Set([
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
  ]);
  const pollingInterval = 30000; // 30 seconds

  try {
    let batchJob = await ai.batches.get({ name: batchJobName });

    while (!completedStates.has(batchJob?.state)) {
      console.log(
        `Current job state: ${batchJob.state}. Waiting ${pollingInterval / 1000} seconds...`,
      );
      await new Promise((resolve) => setTimeout(resolve, pollingInterval));
      batchJob = await ai.batches.get({ name: batchJobName });
    }

    console.log(`Job finished with state: ${batchJob.state}`);
    return batchJob;
  } catch (error) {
    throw new Error(
      `An error occurred while polling job ${batchJobName}: ${error}`,
    );
  }
}

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
