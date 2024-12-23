# Define the source folder containing the images
$SourceFolder = "C:\Users\Teich\github-projects\SWCNN-watermark-removal\data\samples\test-preparation\center"
# Define the output folder for cropped images
$OutputFolder = Join-Path $SourceFolder "RandomCenters"

# Create the output folder if it doesn't exist
if (-not (Test-Path -Path $OutputFolder)) {
    New-Item -ItemType Directory -Path $OutputFolder
}

# Get all "source" images in the folder
$SourceImages = Get-ChildItem -Path $SourceFolder -Filter "*_source*.jpg"

foreach ($SourceImage in $SourceImages) {
    # Derive the corresponding "target" image filename
    $TargetImage = $SourceImage.FullName -replace "_source", "_target"
    
    # Check if the corresponding "target" image exists
    if (Test-Path -Path $TargetImage) {
        # Generate a random offset between -128 and +128 for x and y
        $RandomXOffset = Get-Random -Minimum -128 -Maximum 129
        $RandomYOffset = Get-Random -Minimum -128 -Maximum 129

        # Define the cropped output filenames
        $SourceOutput = Join-Path $OutputFolder ($SourceImage.BaseName + "_randomcenter.jpg")
        $TargetOutput = Join-Path $OutputFolder ((Get-Item $TargetImage).BaseName + "_randomcenter.jpg")

        # Crop the center with random offsets using ImageMagick
        & magick $SourceImage.FullName -gravity center -crop 256x256+$RandomXOffset+$RandomYOffset +repage $SourceOutput
        & magick $TargetImage -gravity center -crop 256x256+$RandomXOffset+$RandomYOffset +repage $TargetOutput

        Write-Host "Processed pair: $($SourceImage.Name) and $(Split-Path $TargetImage -Leaf)"
    } else {
        Write-Host "No matching target image for $($SourceImage.Name)"
    }
}

Write-Host "Processing complete. Cropped images are saved in $OutputFolder"
