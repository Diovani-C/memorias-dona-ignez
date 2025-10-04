// src/cli.ts

import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { existsSync, lstatSync } from "fs";
import { resolve } from "path";

// Import the task creators
import { createRestoreImageTask } from "./src/tasks/RestoreTask.ts";
import { createGenerateAltTextTask } from "./src/tasks/ImageUnderstandingTask.ts";
import { createOptimizeImageTask } from "./src/tasks/OptimizeTask.ts";
import { createUpscaleImageTask } from "./src/tasks/UpscaleTask.ts";
import { createConvertToBwTask } from "./src/tasks/ConvertToBwTask.ts";
import type { TaskFunction } from "./src/types.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .usage("Usage: $0 -t <task> -i <input> -o <output>")
    .option("task", {
      alias: "t",
      describe: "The image processing task to perform",
      choices: [
        "restoration",
        "description",
        "upscale",
        "img2video",
        "compression",
        "bw",
      ] as const,
      demandOption: true,
    })
    .option("input", {
      alias: "i",
      describe: "Input file or folder path",
      type: "string",
      demandOption: true,
    })
    .option("output", {
      alias: "o",
      describe: "Output folder path",
      type: "string",
      demandOption: true,
    })
    .check((argv) => {
      // Custom validation for file paths
      const inputPath = resolve(argv.input);
      const outputPath = resolve(argv.output);

      if (!existsSync(inputPath)) {
        throw new Error(`Input path does not exist: ${inputPath}`);
      }
      if (!existsSync(outputPath)) {
        throw new Error(`Output path does not exist: ${outputPath}`);
      }
      if (!lstatSync(outputPath).isDirectory()) {
        throw new Error(`Output path must be a directory: ${outputPath}`);
      }
      return true;
    })
    .example(
      "$0 -t restoration -i ./old-photos -o ./restored-photos",
      "Restore all images in a folder.",
    )
    .example(
      "$0 -t description -i ./my-image.jpg -o ./output",
      "Generate alt text for a single image.",
    )
    .help("h")
    .alias("h", "help")
    .parseAsync();

  // --- Task Execution ---
  const { task, input, output } = argv;

  // A registry to map task names to their factory functions
  const taskRegistry: Record<string, () => TaskFunction> = {
    restoration: createRestoreImageTask,
    description: createGenerateAltTextTask,
    compression: createOptimizeImageTask,
    upscale: createUpscaleImageTask,
    bw: createConvertToBwTask,
  };

  console.log(`🚀 Starting task: '${task}'...`);
  console.log(`   - Input: ${resolve(input)}`);
  console.log(`   - Output: ${resolve(output)}`);

  try {
    const createTask = taskRegistry[task];
    if (createTask) {
      const taskRunner = createTask();
      await taskRunner({
        inputPath: input,
        outputPath: output,
      });
      console.log(`✅ Task '${task}' completed successfully!`);
    } else {
      // Handle tasks that are defined in options but not yet implemented
      console.warn(
        `🚧 The '${task}' task is not yet implemented. Coming soon!`,
      );
    }
  } catch (error) {
    console.error("\n❌ An error occurred during task execution:");
    if (error instanceof Error) {
      console.error(error.message);
    } else {
      console.error(error);
    }
    process.exit(1); // Exit with an error code
  }
}

// Run the main function
main();
