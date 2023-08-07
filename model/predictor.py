import os
import pickle
import io
import flask

from io import BytesIO
from PIL import Image
import cv2
from flask import send_file
import numpy as np

from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image, resize_image
from collections import namedtuple

from pathlib import Path
import glob

ROOTDIR  = Path(os.getcwd()).parent
HONDALOGO = os.path.join(ROOTDIR,'model','honda_label', 'honda.jpg')
model_path = os.environ['MODEL_PATH']

GAMMA = {
    "0": 2,
    "1": 2,
    "2": 2,
    "3": 2,
    "4": 2,
    
    "7": 0.2,
    "8": 0.2,
    "9": 0.2,
    "10": 0.1,
    "11": 0.04
    
}

def get_img_avg_brightness(img):
    """
    Calculate average image brightness
    Params:
        image: image, basically a 3 dimension np array
    Returns:
        brightness_avg: brightness average
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    brightness_avg = int(np.average(v.flatten()))
    return brightness_avg

BLevel = namedtuple("BLevel", ['brange', 'bval'])

#all possible levels
BRIGHTNESS_LEVELS = [
    BLevel(brange=range(0, 24), bval=0),
    BLevel(brange=range(23, 47), bval=1),
    BLevel(brange=range(46, 70), bval=2),
    BLevel(brange=range(69, 93), bval=3),
    BLevel(brange=range(92, 116), bval=4),
    BLevel(brange=range(115, 140), bval=5),
    BLevel(brange=range(139, 163), bval=6),
    BLevel(brange=range(162, 186), bval=7),
    BLevel(brange=range(185, 209), bval=8),
    BLevel(brange=range(208, 232), bval=9),
    BLevel(brange=range(231, 256), bval=10),
]

def detect_level(avg_brightness):
    """
    Calculate brightness level from average histogram value
    Params:
        h_val: average histogram brightness value
    Returns:
        blevel: brightness level
    """
    avg_brightness = int(avg_brightness)
    for blevel in BRIGHTNESS_LEVELS:
        if avg_brightness in blevel.brange:
            return blevel.bval
    raise ValueError("Brightness Level Out of Range")
    
def adjust_gamma(image, gamma=1.0):
    """
    Adjust image brightness based on gamma provided
    Params:
        image: image, basically a 3 dimension np array
    Returns:
        transformed_img: image with adjusted brightness
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    transformed_img = cv2.LUT(image, table)
    return transformed_img

def blur_image(model, original_image):
    #print(f'\nProcessing {name}')
    threshold = 0.5
    # preprocess image for network
    ori_img = cv2.imread(original_image)
    image = ori_img.copy()
    brightness_val = get_img_avg_brightness(image)
    brightness_lvl = detect_level(brightness_val)
    print(f"brightness average - {brightness_val} | level - {brightness_lvl}")
    
    if str(brightness_lvl) in GAMMA:
        image = adjust_gamma(image, gamma=GAMMA[str(brightness_lvl)])
    
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict(
        np.expand_dims(image, axis=0))
    print(f'confidences:')
    for score, box in zip(scores[0], boxes[0]):
        if score > threshold:
            print(f'{score:.2f}')
            # scale boxes to original image
            box /= scale
            (x1, y1, x2, y2) = (box[0], box[1], box[2], box[3])
            (x1, y1, x2, y2) = np.array((x1, y1, x2, y2)).astype('int16')
            # Used Honda label for blurring
            original_image = Image.open(original_image)
            re_height = y2-y1
            re_width  = x2-x1
            honda_img = Image.open(HONDALOGO)
            # resize honda image to fit into original image
            honda_img = honda_img.resize((re_width,re_height))
            original_image.paste(honda_img,(x1,y1,x2,y2))
    return original_image

def serve_pil_image(predicted_img):
    '''
    Send image file from memory
    image must be PIL type, not numpy array!
    Args:
        Predicted_img: image resulted from model prediction
    Returns:
        send_file object (flask object) to be sent to user
    '''
    # img = Image.fromarray(predicted_img.astype('uint8'))
    img_io = BytesIO()
    predicted_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    result = send_file(img_io, mimetype='image/jpeg')
    return result

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            model_weight = os.path.join(model_path, 'model.h5')
            cls.model = load_model(model_weight, backbone_name='resnet50')
        return cls.model


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    image = None

    # Receive image file 
    #if flask.request.content_type == 'image/jpg' or flask.request.content_type == 'image/jpeg':
    if flask.request.content_type != None:
        image_file = flask.request.files['image'].read()
        nparr = np.fromstring(image_file, np.uint8)
        ori_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return flask.Response(response='Please upload a jpg or jpeg file', status=415, mimetype='text/plain')

    # Do the prediction
    model = ScoringService.get_model()
    #predictions = ScoringService.predict(data)
    blurred_img = blur_image(model, ori_image)
    result = serve_pil_image(blurred_img)
    return result
