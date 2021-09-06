pip install --upgrade tensorflow google-cloud-vision opencv-python
if (Test-Path -Path '.\models') {
    ""
} else {
    git clone https://github.com/tensorflow/models.git

}
Invoke-WebRequest -Uri https://github.com/protocolbuffers/protobuf/releases/download/v3.18.0-rc2/protoc-3.18.0-rc-2-win64.zip -OutFile C:\Users\letic\Downloads\protobuf-all-3.18.0-rc-2.zip
Expand-Archive -Force 'C:\Users\letic\Downloads\protoc-3.18.0-rc-2-win64.zip' 'C:\Users\letic\Google Protoc'
$env:Path += ";C:\Users\letic\Google Protoc\bin"
Set-Location -Path './models/research'
protoc object_detection/protos/*.proto --python_out=.
Copy-Item "object_detection/packages/tf2/setup.py" -Destination .
python -m pip install .
python object_detection/builders/model_builder_tf2_test.py