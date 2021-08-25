## Project title
A little info about your project and/ or overview that explains **what** the project is about.

## Screenshots
Include logo/demo screenshot etc.

## Tech/framework used
<b>Built with</b>
- [Tensorflow Object Detection API](https://electron.atom.io)

## Installation

- Clone this repository

- Install Python: Instructions how to install Python [here](https://www.python.org/downloads/)

- Install Google API: Follow the 'Before you begin' part only: [here](https://cloud.google.com/vision/docs/quickstart-client-libraries).

Once you have the JSON file, move it to the uml folder. Open detection.py and edit this line with the name of your json file:
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./[JSON_FILE_NAME]"

- Install Tensorflow Object Detection API: 
```
# go into the directory in your terminal
cd uml

# Clone the TensorFlow Model Garden repository
git clone https://github.com/tensorflow/models.git

# Install some required libraries and tools.
apt-get install protobuf-compiler python-lxml python-pil
pip install --upgrade tensorflow google-cloud-vision Cython pandas tf-slim lvis

# Compile the Protobuf libraries.
cd 'uml/models/research/'
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

## How to use
Add the handwritten uml diagrams to uml/images.
From the folder uml, run:
```
python3 main.py
```

## API Reference
Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests
Describe and show how to run the tests with code examples.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Leticia Piucco Marques]()
