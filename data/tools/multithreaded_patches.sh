#!/bin/bash

# Check if parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Please install it first:"
    echo "sudo apt-get install parallel  # For Debian/Ubuntu"
    echo "sudo yum install parallel      # For CentOS/RHEL"
    exit 1
fi

# Get parameters from user
read -p "Enter patch width (dimensions are squared): " width
read -p "Enter stride: " stride

# Get CPU info
cpu_cores=$(nproc)
echo "Detected $cpu_cores CPU cores"

# Create patches directory with width and stride in name
patches_dir="patches-${width}-${stride}"
mkdir -p "$patches_dir"

# Function to process a single image file
process_image() {
    local img="$1"
    # Remove any leading ./ from the filename
    img="${img#./}"
    
    # Extract the identifier (everything between _ and .jpg)
    # identifier=$(echo "$img" | sed 's/^[0-9]{3,4}_(.*)\.jpg$/\1/')
    identifier=$(echo "$img" |sed -E 's/^[0-9]{3,4}[A-Za-z]?_(.*)\.jpg$/\1/')
    # Extract the xxx part (first 3 or 4 digits)
    # xxx=$(echo "$img" | sed -E 's/^([0-9]{3,4})[LU]?_.*/\1/')
    xxx=$(echo "$img" | sed -E 's/^([0-9]{3,4}[A-Za-z]?)_.*/\1/')
    
    # Calculate number of patches that will be created
    img_width=$(identify -format "%w" "$img")
    img_height=$(identify -format "%h" "$img")
    patches_per_row=$(( (img_width - width) / stride + 1 ))
    patches_per_col=$(( (img_height - width) / stride + 1 ))
    total_patches=$((patches_per_row * patches_per_col))
    
    echo "Processing $img:"
    echo "- Image dimensions: ${img_width}x${img_height}"
    echo "- Will create $total_patches patches"
    
    count=0
    for ((y=0; y<=img_height-width; y+=stride)); do
        for ((x=0; x<=img_width-width; x+=stride)); do
            # Format patch number with leading zeros (4 digits)
            patch_num=$(printf "%04d" $count)
            
            # Create patch with naming convention:
            # xxx-[# of patch]_[identifier]-[width]-[stride].jpg
            output_file="${patches_dir}/${xxx}-${patch_num}_${identifier}-${width}-${stride}.jpg"
            
            convert "$img" -crop ${width}x${width}+${x}+${y} "$output_file"
            
            # Increment counter
            ((count++))
        done
    done
    
    echo "- Created $count patches"
    echo "-------------------"
}
export -f process_image
export width stride patches_dir

# Find all matching files and process them in parallel
find . -regextype posix-extended -maxdepth 1 -regex '\./[0-9]{3,4}[A-Za-z]?_.*\.jpg' -print0 | \
    sed 's/^\.\///g' | \
    parallel -0 -j "$cpu_cores" --bar process_image {}

echo "Done! All patches are in the $patches_dir directory"