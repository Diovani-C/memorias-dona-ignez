/**
 * @file This script generates descriptive alt text for images to improve web accessibility.
 * It uses the Google Gemini API's image understanding capabilities.
 * - For a single image, it creates a corresponding .txt file with the description.
 * - For multiple images, it runs a batch job and outputs a single CSV file mapping each filename to its generated alt text.
 */

import type { TaskFunction, FileDataType } from "../types.ts";
import { createUserContent } from "@google/genai";
import { join, parse } from "path";
import { Buffer } from "buffer";
import { getGoogleAI, getInputFiles } from "./utils.ts";

// --- Configuration ---
const ALT_TEXT_PROMPT =
  "Descreva esta imagem de forma concisa para acessibilidade na web (texto alternativo). Concentre-se no assunto, ação e contexto essenciais. Não inclua frases como “Uma imagem de” ou “Esta imagem mostra”. A descrição deve ser uma frase única e completa.";
const ALT_TEXT_MODEL = "gemini-2.5-flash-image";

// --- API Initialization ---
const ai = getGoogleAI();

// --- Main Task Function ---

/**
 * Creates a task function for generating image alt text.
 * It automatically handles single or batch processing.
 * @returns {TaskFunction} The function to be executed by the task runner.
 */
export function createGenerateAltTextTask(): TaskFunction {
  return async ({ inputPath, outputPath }) => {
    const inputFiles = await getInputFiles(inputPath);
    if (inputFiles.length === 0) {
      throw new Error("No input files found to process.");
    }

    console.log(
      `Found ${inputFiles.length} file(s) to process for alt text generation.`,
    );

    for (let file of inputFiles) {
      // Await every request response to not reach the API rate limit
      await generateAltText(file, outputPath);
    }
  };
}

// --- Core Logic: Single vs. Batch ---

/**
 * Generates alt text for a single image and saves it to a .txt file.
 * @param {FileDataType} file - The image file to process.
 * @param {string} outputPath - The directory to save the output .txt file.
 */
async function generateAltText(
  file: FileDataType,
  outputPath: string,
): Promise<void> {
  console.log(`Generating alt text for: ${file.name}...`);
  const base64String = Buffer.from(file.data).toString("base64");

  const response = await ai.models.generateContent({
    model: ALT_TEXT_MODEL,
    contents: createUserContent([
      { inlineData: { data: base64String, mimeType: file.mimeType } },
      ALT_TEXT_PROMPT,
    ]),
  });

  const altText = response.text;
  if (!altText) {
    throw new Error(`API failed to generate text for ${file.name}.`);
  }

  // Use the original filename but with a .txt extension
  const parsedPath = parse(file.name);
  const outputFilePath = join(outputPath, `${parsedPath.name}.txt`);

  await Bun.write(outputFilePath, altText.trim());
  console.log(`Successfully saved alt text to ${outputFilePath}`);
}

/**
 * Manages the batch processing workflow to generate alt text for multiple images.
 * @param {FileDataType[]} files - An array of image files.
 * @param {string} outputPath - The directory to save the final results CSV.
 */
// async function generateBatchAltText(
//   files: FileDataType[],
//   outputPath: string,
// ): Promise<void> {
//   console.log("Starting batch alt text generation...");
//
//   const keyToFileNameMap = new Map<string, string>();
//
//   // Step 1: Upload images concurrently.
//   const uploadPromises = files.map(async (file) => {
//     console.log(`Uploading ${file.name}...`);
//     const uploadedFile = await ai.files.upload({
//       file: new Blob([file.data]),
//       config: { mimeType: file.mimeType },
//     });
//     if (!uploadedFile.uri)
//       throw new Error(`Failed to upload file: ${file.name}`);
//
//     const requestKey = crypto.randomUUID();
//     keyToFileNameMap.set(requestKey, file.name);
//
//     return {
//       key: requestKey,
//       request: {
//         contents: createUserContent([
//           createPartFromUri(uploadedFile.uri, uploadedFile.mimeType),
//           ALT_TEXT_PROMPT,
//         ]),
//       },
//     };
//   });
//   const requests: RequestType[] = await Promise.all(uploadPromises);
//   console.log("All files uploaded successfully.");
//
//   // Steps 2-5: Create JSONL, upload, create job, and monitor.
//   const jsonlFilePath = await writeBatchRequestsToFile(requests, outputPath);
//   const uploadedJsonlFile = await ai.files.upload({
//     file: jsonlFilePath,
//     config: { mimeType: "application/jsonl" },
//   });
//   if (!uploadedJsonlFile.name) throw new Error("Error uploading JSONL file.");
//
//   const fileBatchJob = await ai.batches.create({
//     model: ALT_TEXT_MODEL,
//     src: uploadedJsonlFile.name,
//   });
//   if (!fileBatchJob.name) throw new Error("Failed to create batch job.");
//
//   console.log(`Batch job created successfully. Job Name: ${fileBatchJob.name}`);
//   await saveBatchJobInfo(fileBatchJob.name, jsonlFilePath);
//
//   const completedJob = await monitoringBatchJobStatus(ai, fileBatchJob.name);
//
//   // Step 6: Retrieve text results and save them to a single CSV file.
//   if (completedJob.state === "JOB_STATE_SUCCEEDED") {
//     await retrieveAndSaveBatchResults(
//       completedJob,
//       outputPath,
//       keyToFileNameMap,
//     );
//   } else {
//     console.error(
//       `Batch job did not succeed. Final state: ${completedJob.state}`,
//     );
//     if (completedJob.error)
//       console.error(
//         "Error details:",
//         completedJob.error.message || completedJob.error,
//       );
//   }
// }
//
// // --- Helper Functions ---
//
// /**
//  * Retrieves text results from a completed job and saves them to a single JSON file.
//  * @param {any} batchJob - The completed batch job object.
//  * @param {string} outputPath - The directory to save the output JSON.
//  * @param {Map<string, string>} keyToFileNameMap - Map to link keys to filenames.
//  */
// async function retrieveAndSaveBatchResults(
//   batchJob: any,
//   outputPath: string,
//   keyToFileNameMap: Map<string, string>,
// ): Promise<void> {
//   if (!batchJob.dest?.inlinedResponses) {
//     console.log("No inline results found in the batch job.");
//     return;
//   }
//
//   console.log("Processing batch results and creating JSON file...");
//   const results: { filename: string; alt_text: string }[] = [];
//
//   for (const inlineResponse of batchJob.dest.inlinedResponses) {
//     const requestKey = inlineResponse.request.key;
//     const originalFileName = keyToFileNameMap.get(requestKey) || "unknown_file";
//
//     if (inlineResponse.error) {
//       console.error(
//         `Error in response for ${originalFileName}: ${inlineResponse.error.message}`,
//       );
//       results.push({
//         filename: originalFileName,
//         alt_text: "GENERATION_ERROR",
//       });
//       continue;
//     }
//
//     const altText =
//       inlineResponse.response?.text()?.trim() || "NO_TEXT_GENERATED";
//     results.push({ filename: originalFileName, alt_text: altText });
//   }
//
//   // Convert results array to a pretty-printed JSON string
//   const jsonContent = JSON.stringify(results, null, 2);
//   const outputJsonPath = join(outputPath, "alt_text_results.json");
//
//   await writeFile(outputJsonPath, jsonContent);
//   console.log(
//     `Successfully wrote ${results.length} results to ${outputJsonPath}`,
//   );
// }
