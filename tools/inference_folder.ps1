# Function to check if inside a virtual environment
function Check-VirtualEnv {
    $VenvPath = Get-Item -Path ".venv/Scripts/activate" -ErrorAction SilentlyContinue
    if ($null -ne $env:VIRTUAL_ENV) {
        Write-Host "Virtual environment detected: $env:VIRTUAL_ENV"
        return $true
    } elseif ($null -ne $VenvPath) {
        Write-Host "No virtual environment detected. Activating .venv..."
        & .venv/Scripts/activate
        if ($null -ne $env:VIRTUAL_ENV) {
            Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
            return $true
        } else {
            Write-Host "Failed to activate virtual environment." -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "No virtual environment detected and .venv does not exist." -ForegroundColor Red
        return $false
    }
}

# Check virtual environment
if (-not (Check-VirtualEnv)) {
    Write-Host "Terminating script. Please ensure you have a virtual environment set up in .venv." -ForegroundColor Red
    exit
}

# Define variables
$InputDirectory = "data/test/watermarked"
$OutputBaseDirectory = "output/inference_runs"

# Prompt user for the model path
$ModelPath = Read-Host "Enter the model path (e.g., output/models/HN_per_L1_n2n_029_best.pth)"

# Generate a new folder in the output directory with the current timestamp
$Timestamp = Get-Date -Format "yy-MM-dd-HH-mm-ss"
$OutputDir = Join-Path $OutputBaseDirectory $Timestamp
New-Item -ItemType Directory -Path $OutputDir | Out-Null

# Get all jpg files from the input directory
$JpgFiles = Get-ChildItem -Path $InputDirectory -Filter "*.jpg"

# TODO: Replace with a python implementation so we don't need to load the model each time
# Iterate over each jpg file and perform the Python command
foreach ($File in $JpgFiles) {
    # Get the output file path
    $OutputFileName = $File.BaseName + "_output.jpg"
    $OutputFilePath = Join-Path $OutputDir $OutputFileName

    # Construct the Python command
    $PythonCommand = "python inference.py --model_path `"$ModelPath`" --input_image `"$($File.FullName)`" --output_image `"$OutputFilePath`""

    # Execute the command
    Write-Host "Processing $($File.Name)..."
    Invoke-Expression $PythonCommand
}

Write-Host "Inference completed. Results are saved in $OutputDir"
