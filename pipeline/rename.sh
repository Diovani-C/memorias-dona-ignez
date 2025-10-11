#!/bin/bash

# A simple script to sequentially rename all image files in a directory.

# Initialize a counter
i=1

# Loop through all files with common image extensions (case-insensitive)
for file in *.{jpg,jpeg,png,gif,JPG,JPEG,PNG,GIF}; do
  # Check if files of this type exist to avoid errors
  if [ -f "$file" ]; then
    # Format the counter with a leading zero (e.g., 01, 02, 03)
    num=$(printf "%02d" $i)

    # Get the file's original extension
    ext="${file##*.}"

    # Set the new name, keeping the original extension
    new_name="fragment-$num.$ext"

    # Rename the file
    echo "Renaming '$file' to '$new_name'"
    mv -- "$file" "$new_name"

    # Increment the counter
    i=$((i + 1))
  fi
done

echo "✅ Renaming complete!"
