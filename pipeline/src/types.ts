/**
 * Represents the data of a file read from the input path.
 */
export interface FileDataType {
  name: string; // The name of the file (e.g., 'photo.jpg')
  data: ArrayBuffer; // The raw file data as a Buffer
  mimeType: string; // The MIME type of the file (e.g., 'image/jpeg')
}

/**
 * Defines the shape of the arguments passed to a task function.
 */
export interface TaskArguments {
  inputPath: string;
  outputPath: string;
}

/**
 * Defines the function signature for any task that can be executed by the CLI.
 */
export type TaskFunction = (args: TaskArguments) => Promise<void>;
