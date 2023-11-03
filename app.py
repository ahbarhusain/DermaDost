from flask import render_template, Flask, request
import random
import numpy as np

from tensorflow.keras.applications import EfficientNetB3
import numpy as np
from tensorflow.keras.models import model_from_json,load_model

from keras import backend as K
import cv2
# Chatbot imports
import json
import pickle
import nltk
import wikipedia

import numpy as np
import random
#####################


app = Flask(__name__)
# Load your image classification model and setup variables
eff3_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
for layer in eff3_model.layers:
    layer.trainable = False

model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)

def get_class(image_path):
    try:
        img = cv2.imread(image_path)
    except BaseException:
        return 'false'
    else:
        img = cv2.resize(img, (180, 180))
        img = np.array(img)
        img = img.reshape(1, 180, 180, 3)
        x_t = eff3_model.predict(img)
        x_t = x_t.reshape(1, -1)

        # Predict the result
        result = loaded_model.predict(x_t)
        # Define class names based on your dataset
        class_names = ['Acnes', 'Healthy', 'Vitiligo', 'Fungal Infections',
                       'Melanoma Skin Cancer and Moles', 'Eczema']
        predicted_class_index = np.argmax(result)
        predicted_class_name = class_names[predicted_class_index]
        print(predicted_class_name)
        return predicted_class_name


@app.route('/')
def index():
    return render_template('index.html', title='Home')


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = 'static/data/' + f.filename
        f.save(path)
        disease = get_class(path)
        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=100, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)


