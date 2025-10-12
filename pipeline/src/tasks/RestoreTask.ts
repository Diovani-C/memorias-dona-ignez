/**
 * @file This script provides functions to restore old photographs using the Google Gemini API.
 * It supports processing a single image or batch-processing multiple images.
 * - Single images are processed via a standard `generateContent` API call.
 * - Multiple images are processed using the more efficient Batch API, which involves:
 * 1. Uploading each image to the Google AI File Service.
 * 2. Creating a JSONL file with requests pointing to the uploaded image URIs.
 * 3. Uploading the JSONL file and creating a batch job.
 * 4. Monitoring the batch job until completion.
 * 5. Retrieving and saving the restored images.
 */

import type { TaskFunction, FileDataType } from "../types.ts";
import { createUserContent } from "@google/genai";
import { join } from "path";
import { Buffer } from "buffer";
import { getGoogleAI, getInputFiles } from "./utils.ts";

// --- Configuration ---
const RESTORE_TASK_PROMPT = `You are a photo restoration and enhancement specialist. Your task is to meticulously restore the provided old photograph to its highest possible quality, making it look as if it was captured recently with professional equipment, while preserving the original style, lighting, and atmosphere without adding new elements or changing the identity of the people.
Repair Physical Damage: Systematically remove all forms of physical degradation. This includes repairing scratches, tears, creases, dust spots, chemical stains, and any other imperfections and missing details that detract from the original scene.
Correct Tonal and Color Issues: 
	If the image is black and white or sepia-toned, produce realistically colorized version, carefully colorize the image, keeping skin tones, clothing, objects, and background realistic.
	If the image originally had color but has faded or discolored, restore its original vibrancy, correct any color shifts, and enhance overall color accuracy and saturation to a natural level.
Enhance Clarity and Detail: Sharpen soft or blurry areas without creating artifacts. Recover lost textures and fine details in faces, clothing, hair, objects, and backgrounds. Improve overall image clarity and definition.`;

const RESTORE_TASK_MODEL = "gemini-2.5-flash-image";

// --- API Initialization ---
const ai = getGoogleAI();

// --- Main Task Function ---

/**
 * Creates a task function for restoring images.
 * It dynamically chooses between single or batch processing based on the number of input files.
 * @returns {TaskFunction} The function to be executed by the task runner.
 */
export function createRestoreImageTask(): TaskFunction {
  return async ({ inputPath, outputPath }) => {
    const inputFiles = await getInputFiles(inputPath);
    if (inputFiles.length === 0) {
      throw new Error("No input files found to process.");
    }

    console.log(`Found ${inputFiles.length} file(s) to process.`);

    for (let file of inputFiles) {
      // Await every request response to not reach the API rate limit
      await restoreImage(file, outputPath);
    }
  };
}

// --- Core Logic: Single vs. Batch ---

/**
 * Restores a single image using a standard API call and saves the result.
 * @param {FileDataType} file - The file to process, containing its name, data, and MIME type.
 * @param {string} outputPath - The directory where the restored image will be saved.
 */
async function restoreImage(
  file: FileDataType,
  outputPath: string,
): Promise<void> {
  console.log(`Restoring image: ${file.name}...`);
  const base64String = Buffer.from(file.data).toString("base64");

  const response = await ai.models.generateContent({
    model: RESTORE_TASK_MODEL,
    contents: createUserContent([
      { inlineData: { data: base64String, mimeType: file.mimeType } },
      RESTORE_TASK_PROMPT,
    ]),
  });

  // Error handling for API response
  if (!response.candidates || response.candidates.length === 0) {
    throw new Error(
      `API returned no candidates for the restored image. ${response.data}`,
    );
  }

  const imagePart = response.candidates?.[0]?.content?.parts?.find(
    (p) => p.inlineData,
  );

  if (imagePart?.inlineData) {
    const imageData = imagePart.inlineData.data;
    const responseBuffer = Buffer.from(imageData, "base64");
    const outputFilePath = join(outputPath, `restored_${file.name}`);

    await Bun.write(outputFilePath, responseBuffer);
  } else {
    throw new Error(
      `Could not find image data in the API response. ${response.data}`,
    );
  }
}

/**
 * Manages the entire batch processing workflow for restoring multiple images.
 * @param {FileDataType[]} files - An array of files to process.
 * @param {string} outputPath - The directory where restored images will be saved.
 */
// async function restoreBatchImages(
//   files: FileDataType[],
//   outputPath: string,
// ): Promise<void> {
//   console.log("Starting batch image restoration...");
//
//   // Use a map to associate a unique key with each original filename.
//   const keyToFileNameMap = new Map<string, string>();
//
//   // Step 1: Upload all images concurrently to the File API.
//   const uploadPromises = files.map(async (file) => {
//     console.log(`Uploading ${file.name}...`);
//     const uploadedFile = await ai.files.upload({
//       file: new Blob([file.data]),
//       config: { mimeType: file.mimeType },
//     });
//
//     if (!uploadedFile.uri || !uploadedFile.mimeType) {
//       throw new Error(`Failed to upload file to Gemini cloud: ${file.name}`);
//     }
//
//     const requestKey = crypto.randomUUID(); // Use standard crypto API
//     keyToFileNameMap.set(requestKey, file.name);
//
//     return {
//       key: requestKey,
//       request: {
//         contents: createUserContent([
//           createPartFromUri(uploadedFile.uri, uploadedFile.mimeType),
//           RESTORE_TASK_PROMPT,
//         ]),
//       },
//     };
//   });
//
//   const requests: RequestType[] = await Promise.all(uploadPromises);
//   console.log("All files uploaded successfully.");
//
//   // Step 2: Write requests to a JSONL file.
//   const jsonlFilePath = await writeBatchRequestsToFile(requests, outputPath);
//
//   // Step 3: Upload the JSONL file to start the batch job.
//   const uploadedJsonlFile = await ai.files.upload({
//     file: jsonlFilePath,
//     config: { mimeType: "application/jsonl" }, // Correct MIME type
//   });
//
//   if (!uploadedJsonlFile.name) {
//     throw new Error("Error uploading JSONL file to File API.");
//   }
//
//   // Step 4: Create and start the batch job.
//   const fileBatchJob = await ai.batches.create({
//     model: RESTORE_TASK_MODEL,
//     src: uploadedJsonlFile.name,
//   });
//
//   if (!fileBatchJob.name) {
//     throw new Error("Failed to create batch job.");
//   }
//
//   console.log(`Batch job created successfully. Job Name: ${fileBatchJob.name}`);
//   await saveBatchJobInfo(fileBatchJob.name, jsonlFilePath);
//
//   // Step 5: Monitor the job until it completes.
//   const completedJob = await monitoringBatchJobStatus(ai, fileBatchJob.name);
//
//   // Step 6: Retrieve results if the job succeeded.
//   if (completedJob.state === "JOB_STATE_SUCCEEDED") {
//     await retrieveBatchJobResults(completedJob, outputPath, keyToFileNameMap);
//   } else {
//     console.error(
//       `Batch job did not succeed. Final state: ${completedJob.state}`,
//     );
//     if (completedJob.error) {
//       console.error(
//         "Error details:",
//         completedJob.error.message || JSON.stringify(completedJob.error),
//       );
//     }
//   }
// }
//
// /**
//  * Retrieves and processes the results from a completed batch job.
//  * @param {any} batchJob - The completed batch job object from the API.
//  * @param {string} outputPath - The directory to save the output files.
//  * @param {Map<string, string>} keyToFileNameMap - A map to link response keys to original filenames.
//  */
// async function retrieveBatchJobResults(
//   batchJob: any,
//   outputPath: string,
//   keyToFileNameMap: Map<string, string>,
// ): Promise<void> {
//   if (batchJob.dest?.inlinedResponses) {
//     console.log("Processing inline results...");
//     let successCount = 0;
//
//     for (const inlineResponse of batchJob.dest.inlinedResponses) {
//       const requestKey = inlineResponse.request.key;
//       const originalFileName = keyToFileNameMap.get(requestKey);
//
//       if (!originalFileName) {
//         console.error(
//           `Error: Could not find original filename for request key: ${requestKey}`,
//         );
//         continue;
//       }
//
//       if (inlineResponse.error) {
//         console.error(
//           `Error processing ${originalFileName}: ${inlineResponse.error.message}`,
//         );
//         continue;
//       }
//
//       // Find the image data in the response
//       const imagePart =
//         inlineResponse.response?.candidates?.[0]?.content?.parts?.find(
//           (p) => p.inlineData,
//         );
//
//       if (imagePart) {
//         const imageData = imagePart.inlineData.data;
//         const outputBuffer = Buffer.from(imageData, "base64");
//         const outputFilePath = join(outputPath, `restored_${originalFileName}`);
//
//         await writeFile(outputFilePath, outputBuffer);
//         console.log(`Saved restored image: ${outputFilePath}`);
//         successCount++;
//       } else {
//         console.error(
//           `No image data found in response for ${originalFileName}.`,
//         );
//       }
//     }
//     console.log(
//       `Finished processing results: ${successCount} successful restorations.`,
//     );
//   } else {
//     console.log(
//       "No inline results found. Check job destination settings if you expected a file output.",
//     );
//   }
// }
