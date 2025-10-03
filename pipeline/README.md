# Tasks

- [x] Build a CLI that accepts a folder or file, process it and outputs to a folder
- [x] Add different actions that the CLI can perform
  - [x] Image restoration
  - [x] Image description
  - [ ] Upscale
  - [x] Image size Optimization
  - [ ] Image to video
  - [ ] Face recognition

## Batch API only works with paid plan

- [x] Add the batch API
  - [x] Send all files to the server and store their information in json file
  - [x] Send a batch task and store the batch information in a json file
  - [x] Task to consult a task state using a json file
  - [x] Retrieve the data from task and delete the files on the server

## Command to run the CLI

```bash
bun index.ts -i ./input -o ./output -t restoration
```
