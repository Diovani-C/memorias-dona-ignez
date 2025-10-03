/**
 * @file Task to upscale images using UpscalerJS.
 * It loads a pre-trained model to increase the resolution of input images.
 */

import Upscaler from "upscaler";
import x4 from "@upscalerjs/esrgan-thick/4x";
import * as tf from "@tensorflow/tfjs-node";
import { join, parse } from "path";
import type { TaskFunction, FileDataType } from "../types.js";
import { getInputFiles } from "./utils.js";

/**
 * Upscales a single image using UpscalerJS and saves the result.
 * @param {FileDataType} file - The file to upscale.
 * @param {string} outputPath - The directory to save the upscaled image.
 */
async function upscaleSingleImage(
  file: FileDataType,
  outputPath: string,
  upscaler: InstanceType<typeof Upscaler>,
): Promise<void> {
  console.log(`   -> Upscaling image: ${file.name}...`);

  try {
    // Decode the image buffer into a tensor
    const tensor = tf.node.decodeImage(
      new Uint8Array(file.data),
      3,
    ) as tf.Tensor3D; // 3 channels for RGB

    // Perform the upscaling
    const upscaledTensor = (await upscaler.upscale(tensor, {
      output: "tensor", // Request output as a tensor
      patchSize: 64, // Optimal for performance on larger images
      padding: 2, // Optional: helps with border artifacts
    })) as tf.Tensor3D; // Cast back to Tensor3D

    // Encode the upscaled tensor back to an image buffer (JPEG for this example)
    // Note: You could also choose PNG, but JPEG is often better for photos.
    // Quality can be adjusted.
    const upscaledBuffer = await tf.node.encodeJpeg(upscaledTensor, "", 90);

    // Clean up tensors
    tf.dispose([tensor, upscaledTensor]);

    const originalName = parse(file.name).name;
    const outputFilePath = join(outputPath, `${originalName}_upscaled_4x.jpg`); // Save as JPEG

    await Bun.write(outputFilePath, upscaledBuffer);
    console.log(`   -> Saved upscaled image to ${outputFilePath}`);
  } catch (error) {
    console.error(`Failed to upscale ${file.name}. Skipping.`, error);
  }
}

/**
 * Creates a task function for upscaling images.
 * @returns {TaskFunction} The function to be executed by the CLI.
 */
export function createUpscaleImageTask(): TaskFunction {
  return async ({ inputPath, outputPath }) => {
    const inputFiles = await getInputFiles(inputPath);

    if (inputFiles.length === 0) {
      throw new Error("No input files found to process.");
    }

    const upscaler = new Upscaler({
      model: x4,
    });

    console.log(`Found ${inputFiles.length} file(s) to upscale.`);

    for (let file of inputFiles) {
      await upscaleSingleImage(file, outputPath, upscaler);
    }
  };
}
