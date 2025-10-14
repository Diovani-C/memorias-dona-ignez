#!/bin/bash

# --- CONFIGURATION ---
# The length of the random string in the new filename.
RANDOM_LENGTH=8

# --- SCRIPT ---
# Check if a directory path was provided as an argument.
if [ -z "$1" ]; then
  echo "🚫 Error: No directory specified."
  echo "   Usage: ./rename_files.sh /path/to/your/folder"
  exit 1
fi

TARGET_DIR="$1"

# Check if the provided path is actually a directory.
if [ ! -d "$TARGET_DIR" ]; then
  echo "🚫 Error: '$TARGET_DIR' is not a valid directory."
  exit 1
fi

# Get a count of the files to determine padding for the number prefix.
# We use `find` to count only the files (-type f) in the target directory.
file_count=$(find "$TARGET_DIR" -maxdepth 1 -type f | wc -l)

# If no files are found, exit gracefully.
if [ "$file_count" -eq 0 ]; then
  echo "No files found in '$TARGET_DIR'. Nothing to do."
  exit 0
fi

padding=$(echo -n "$file_count" | wc -c)
counter=1

echo "Found $file_count files in '$TARGET_DIR'. Starting rename process..."

# Find all files in the target directory, sort them, and loop through each one.
# IFS= and -r prevent issues with filenames containing spaces or special characters.
find "$TARGET_DIR" -maxdepth 1 -type f | sort | while IFS= read -r filepath; do
  # Get just the filename from the full path (e.g., "/path/to/photo.jpg" -> "photo.jpg").
  filename=$(basename -- "$filepath")

  # Get the file extension.
  extension="${filename##*.}"

  # Generate a random string using /dev/urandom for better randomness.
  random_string=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w $RANDOM_LENGTH | head -n 1)

  # Create the new filename. `printf` handles adding the leading zeros to the counter.
  if [ "$filename" = "$extension" ]; then
    # Handle files with no extension.
    new_name=$(printf "%0${padding}d_%s" "$counter" "$random_string")
  else
    new_name=$(printf "%0${padding}d_%s.%s" "$counter" "$random_string" "$extension")
  fi

  # Construct the full path for the new file inside the target directory.
  new_filepath="$TARGET_DIR/$new_name"

  # Rename the file. The quotes handle spaces in names.
  mv -- "$filepath" "$new_filepath"

  echo "Renamed '$filename' to '$new_name'"

  # Increment the counter for the next file.
  counter=$((counter + 1))
done

echo "----------------------------"
echo "✅ Done! All files in '$TARGET_DIR' have been renamed."
