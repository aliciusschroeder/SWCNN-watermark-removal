#!/bin/bash

# Function to check image dimensions
check_dimensions() {
    file="$1"
    dimensions=$(identify -format "%wx%h" "$file")

    if [[ "$dimensions" != "256x256" ]]; then
        echo "File '$file' is not 256x256. Dimensions: $dimensions"
    fi
}

export -f check_dimensions

# Use GNU Parallel to run the function on all .jpg files
find . -name "*.jpg" | parallel -j$(nproc) check_dimensions
