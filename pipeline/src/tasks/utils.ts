/**
 * @file Shared utility functions for interacting with the Google Gemini API.
 * This includes API client initialization, batch file processing, and job monitoring.
 */

import { GoogleGenAI, type Content } from "@google/genai";
import { resolve, join } from "path";
import { stat, appendFile, writeFile } from "fs/promises";

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
