#!/bin/bash

# Prompt the user for input
read -p "Enter the directory you want to navigate to: " target_dir

# Check if the input is a valid directory
if [ -d "$target_dir" ]; then
    # Change directory
    cd "$target_dir" || exit
    echo "Changed directory to: $(pwd)"
else
    echo "Error: '$target_dir' is not a valid directory."
    exit 1
fi

mkdir clean
mkdir watermarked
mv *target*.jpg watermarked
mv *source*.jpg clean
