$sourceFiles = @(
    "target\debug\decoder.dll",
    "target\debug\decoder.dll.exp",
    "target\debug\decoder.dll.lib"
)
$destinationPath = "..\..\dotnet\infer_onnx\"

# Copy each file to the destination, overwriting existing files
foreach ($file in $sourceFiles) {
    Copy-Item -Path $file -Destination $destinationPath -Force
}