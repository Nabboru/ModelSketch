## Project title
A little info about your project and/ or overview that explains **what** the project is about.

## Screenshots
Include logo/demo screenshot etc.

## Tech/framework used
<b>Built with</b>
- [Tensorflow Object Detection API](https://electron.atom.io)


## Installation
- Python: Instructions how to install Python [here]

pip install --ignore-installed --upgrade tensorflow==2.5.0

- Install Google API
You can follow the 'Before you begin' and 'Installation' [here](https://cloud.google.com/vision/docs/quickstart-client-libraries). No need to follow '' because that is alreday implemented in the code.

- Install Tensorflow Object Detection API
cd Repository
git clone https://github.com/tensorflow/models.git
apt-get install protobuf-compiler python-lxml python-pil
pip install Cython pandas tf-slim lvis opencv-python
pip install --upgrade google-cloud-vision

%cd '/content/gdrive/My Drive/TensorFlow/models/research/'
protoc object_detection/protos/*.proto --python_out=.

## How to use
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## API Reference
Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests
Describe and show how to run the tests with code examples.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Yourname]()
