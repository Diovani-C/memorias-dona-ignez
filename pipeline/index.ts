import { GoogleGenAI } from "@google/genai";
import mime from "mime";
import { writeFile, readFileSync, readdir } from "fs";
import { join } from "path";

const INPUT_FOLDER_PATH = join(__dirname, "input");
const OUTPUT_FOLDER_PATH = join(__dirname, "output");

function saveBinaryFile(fileName: string, content: Buffer) {
  writeFile(fileName, content, "utf8", (err) => {
    if (err) {
      console.error(`Error writing file ${fileName}:`, err);
      return;
    }
    console.log(`File ${fileName} saved to file system.`);
  });
}

function readImageFile(imgPath: string) {
  let mimeType: string;
  if (imgPath.includes(".jpeg") || imgPath.includes(".jpg"))
    mimeType = "image/jpeg";
  else mimeType = "image/png";

  const imagePath = imgPath;
  const imageData = readFileSync(imagePath);
  return {
    data: imageData.toString("base64"),
    mimeType,
  };
}

async function main() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (apiKey == undefined) throw "API key not found";

  const ai = new GoogleGenAI({
    apiKey,
  });
  const config = {
    responseModalities: ["IMAGE", "TEXT"],
  };
  const model = "gemini-2.5-flash-image-preview";
  const contents = [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            data: ``,
            mimeType: `image/jpeg`,
          },
        },
        {
          text: `Restore this old photograph. Please remove scratches, dust, and fix the faded colors. Improve the overall clarity and sharpness.`,
        },
      ],
    },
  ];

  const response = await ai.models.generateContentStream({
    model,
    config,
    contents,
  });

  let fileIndex = 0;

  for await (const chunk of response) {
    if (
      !chunk.candidates ||
      !chunk.candidates[0].content ||
      !chunk.candidates[0].content.parts
    ) {
      continue;
    }

    if (chunk.candidates?.[0]?.content?.parts?.[0]?.inlineData) {
      const fileName = `ENTER_FILE_NAME_${fileIndex++}`;
      const inlineData = chunk.candidates[0].content.parts[0].inlineData;
      const fileExtension = mime.getExtension(inlineData.mimeType || "");
      const buffer = Buffer.from(inlineData.data || "", "base64");
      saveBinaryFile(`${fileName}.${fileExtension}`, buffer);
    } else {
      console.log(chunk.text);
    }
  }
}

main();
