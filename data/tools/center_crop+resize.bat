@echo off

:: Create output directories if they don't exist
set outputDirCenter=patches-256-centeronly
set outputDir90=patches-256-90-centeronly
set outputDir110=patches-256-110-centeronly

if not exist "%outputDirCenter%" mkdir "%outputDirCenter%"
if not exist "%outputDir90%" mkdir "%outputDir90%"
if not exist "%outputDir110%" mkdir "%outputDir110%"

:: Process each .jpg file in the current directory
for %%f in (*.jpg) do (
    echo Processing %%f...

    :: Original cropping
    magick "%%f" -gravity center -crop 256x256+0+0 +repage "%outputDirCenter%\%%~nf-256-centeronly.jpg"

    :: Resize to 90% and crop
    magick "%%f" -resize 90%% -gravity center -crop 256x256+0+0 +repage "%outputDir90%\%%~nf-256-90-centeronly.jpg"

    :: Resize to 110% and crop
    magick "%%f" -resize 110%% -gravity center -crop 256x256+0+0 +repage "%outputDir110%\%%~nf-256-110-centeronly.jpg"
)

echo All images have been processed and saved in the respective folders:
echo - %outputDirCenter%
echo - %outputDir90%
echo - %outputDir110%
pause
