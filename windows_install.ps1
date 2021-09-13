pip install --upgrade tensorflow google-cloud-vision opencv-python
if (Test-Path -Path '.\models') {
    ""
} else {
    git clone https://github.com/tensorflow/models.git

}

$username = [Environment]::UserName
Write-Output $username 
Invoke-WebRequest -Uri https://github.com/protocolbuffers/protobuf/releases/download/v3.18.0-rc2/protoc-3.18.0-rc-2-win64.zip -OutFile C:\Users\$username\Downloads\protobuf-all-3.18.0-rc-2.zip
Expand-Archive -Force "C:\Users\$username\Downloads\protoc-3.18.0-rc-2-win64.zip" "C:\Users\$username\Google Protoc"
$env:Path += ";C:\Users\$username\Google Protoc\bin"
Set-Location -Path './models/research'
protoc object_detection/protos/*.proto --python_out=.
Copy-Item "object_detection/packages/tf2/setup.py" -Destination .
python -m pip install .
python object_detection/builders/model_builder_tf2_test.py