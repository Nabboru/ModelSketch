import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
import os
from google.cloud import vision
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import copy
import glob

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

img =  glob.glob("./images/*")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./gleaming-vision-321216-443860317142.json"


# Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap(
    "./label_map.pbtxt", use_display_name=True)

PATH_TO_SAVED_MODEL="./uml_model"

print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

print('Done!')

class SimpleClass():
    def __init__(self, box, text):
        self.box = box
        self.title = ''.join(text)
        self.is_parent = False
        self.inheritance = None
        self.associations = []
        
    def __str__(self):
        string = '\n'
        if self.is_parent:
            string += 'abstract '

        string += (f'class {self.title} ')

        if self.inheritance:
            try:
                string += (f'extends {self.inheritance.parent.title} ')
            except:
                string += (f'extends ?? ')
        
        string += '{ }'

        if self.associations:
            for a in self.associations:
                try:
                    string += (f'\n{{ reference: {a.get_other_class(self).title} }}')
                except:
                    string += (f'\n{{ reference: ??')
        return string

    def add_association(self, association):
        self.associations.append(association)
    
    def add_inheritance(self, inheritance):
        self.inheritance = inheritance
    
    def set_parent(self):
        self.is_parent = True
        
    
class ClassWithAttributes():
    def __init__(self, box, text):
        self.box = box
        self.title = text[0]
        self.text = text
        self.is_parent = False
        self.inheritance = None
        self.associations = []

    def __str__(self):
        string = '\n'
        
        # Check if there is inheritances related to this class
        if self.is_parent:
            string += 'abstract '

        string += (f'class {self.title} ')

        if self.inheritance:
            try:
                string += (f'extends {self.inheritance.parent.title} ')
            except:
                string += (f'extends ?? ')
        string += '{'
        i = 1
        while i < len(self.text):
            if self.text[i] != ':':
                try:
                    if self.text[i+1] == ':' and self.text[i+2]:
                        string += (f'\n\t{self.text[i]}: {self.text[i+2]};')
                        i += 2
                except IndexError:
                    string += (f'\n\t{self.text[i]};')
            i += 1
        
        string += ('\n}')

        # Check if there are associations related to this class
        # and print accordingly
        if self.associations:
            for a in self.associations:
                try:
                    string += (f'\n{{ reference: {a.get_other_class(self).title} }}')
                except:
                    string += (f'\n{{ reference: ??')

        return string

    def add_association(self, association):
        self.associations.append(association)
    
    def add_inheritance(self, inheritance):
        self.inheritance = inheritance

    def set_parent(self):
        self.is_parent = True
        
class Inheritance():
    def __init__(self, box):
      self.box = box
      self.parent = None
      self.children = []
      
    def add_parent(self, parent):
        self.parent = parent
    def add_child(self, child):
        self.children.append(child)

class Association():
    def __init__(self, box):
        self.box = box
        self.classes = []
    
    def get_other_class(self, in_class):
        if self.classes[0] == in_class:
            return self.classes[1]
        return self.classes[0]
    
    def add_class(self, in_class):
        self.classes.append(in_class)
        # Remove the class with smallest intersect area in case more than two 
        # classes intersect with the association
        if len(self.classes) > 2:
            int1 = intersection(self.classes[0].box, self.box)
            int2 = intersection(self.classes[1].box, self.box)
            int3 = intersection(self.classes[2].box, self.box)
            inter_list = [int1, int2, int3]
            self.classes.pop(inter_list.index(min(inter_list)))

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    
    Args: 
        path: the file path to the image
    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def detect_document(img):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    content = cv2.imencode('.jpg', img)[1].tostring()

    image = vision.Image(content=content)
    
    response = client.document_text_detection(image=image)

    useless_text = ['\"']
    text = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols if symbol.text not in useless_text])
                    text.append(word_text)

    if response.error.message:
        raise Exception(
        f'{response.error.message}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors')
    return text

def intersection_top(box1, box2):
    """
    Args:
        box1:
        box2:
    Returns:
        Boolean
    """
    new_box1 = copy.deepcopy(box1)
    diff = abs(new_box1[0] - new_box1[2]) / 2.0
    new_box1[2] = new_box1[2] - diff
    return intersection(new_box1, box2)

def intersection(box1, box2):
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2
    min_ymax = min(y_max1, y_max2)
    max_ymin = max(y_min1, y_min2)
    intersect_heights = max(0, min_ymax - max_ymin)
    min_xmax = min(x_max1, x_max2)
    max_xmin = max(x_min1, x_min2)
    intersect_widths = max(0, min_xmax - max_xmin)
    return intersect_heights * intersect_widths

def detect(image_path):
    image_np = load_image_into_numpy_array(image_path)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
        
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)

    # Detect the image
    og_detections = detect_fn(input_tensor)

    # Convert output to numpy arrays, take index [0] to remove the batch 
    # dimension and filter out detections in order to get 
    # only boxes, classes and scores
    key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
    num_detections = int(og_detections.pop('num_detections'))
    detections = {key:value[0,:num_detections].numpy()
                for key,value in og_detections.items() if key in key_of_interest}

    # The detection classes should be integers.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
    # Filter detection if the confidence threshold of it is smaller than 80%
    for key in key_of_interest:
        scores = detections['detection_scores']
        current_array = detections[key]
        filtered_current_array = current_array[scores > 0.8]
        detections[key] = filtered_current_array
    return detections

def crop_box(coordinates, height, width, image):
    ymin = int(coordinates[0] * height)
    xmin = int(coordinates[1] * width)
    ymax = int(coordinates[2] * height)
    xmax = int(coordinates[3] * width)
    return image[ymin:ymax, xmin:xmax, :]


def main():
    for image_path in img:
        print(f'\nDectection for {image_path}')
        print('-----------------------------------------------------------------')
        detections = detect(image_path)
        image_np = load_image_into_numpy_array(image_path)
        image_np_with_detections=image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=.8,      #min prediction threshold
                agnostic_mode=False,
                line_thickness=8)
        plt.figure(figsize=(18, 18))
        plt.imshow(image_np_with_detections)
        plt.show()

        image = cv2.imread(image_path)
        height, width, channels = image.shape
        classes = []
        associations = []
        inheritance = []
        for i in range(len(detections['detection_classes'])):
            box_class = detections['detection_classes'][i]
            box_cropped = crop_box(detections['detection_boxes'][i], height, width, image)
            if box_class == 1:
                text = detect_document(box_cropped)
                classes.append(SimpleClass(detections['detection_boxes'][i], text))
            elif box_class == 2:
                text = detect_document(box_cropped)
                classes.append(ClassWithAttributes(detections['detection_boxes'][i], text))
            elif box_class == 3:
                inheritance.append(Inheritance(detections['detection_boxes'][i]))
            elif box_class == 4:
                text = detect_document(box_cropped)
                associations.append(Association(detections['detection_boxes'][i]))
        
        for i in inheritance:
            for c in classes:
                if intersection_top(i.box, c.box) > 0.0:
                    i.add_parent(c)
                    c.set_parent()
                elif intersection(i.box, c.box) > 0.0:
                    i.add_child(c)
                    c.add_inheritance(i)

        for a in associations:
            for c in classes:
                if intersection(a.box, c.box) > 0.0:
                    a.add_class(c)
                    c.add_association(a)

        for c in classes:
            print(c)
main()
