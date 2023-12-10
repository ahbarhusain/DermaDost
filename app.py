from flask import render_template, Flask, request
import random
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
from tensorflow.keras.models import model_from_json,load_model
from ultralytics import YOLO
from keras import backend as K
import cv2
# Chatbot imports
import json
import pickle
import nltk
import wikipedia
import os
import numpy as np
import random
import shutil
from roboflow import Roboflow

rf = Roboflow(api_key="6F3CyN0u3MAYiBSY6apX")
project = rf.workspace().project("acne-yolo")
yolo_model = project.version(1).model
#yolo_model = YOLO('ComputerVision/yolomodel/detect/train6/weights/best.pt')
emails = ["dermadost@gmail.com"]
passwords = ["12345678"]
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
    img = Image.open(image_path)  # Use Pillow to open the image
  except BaseException:
    return 'false'
  else:
    try:
      img = img.resize((180, 180))
      img = np.array(img)
      img = img.reshape(1, 180, 180, 3)
      x_t = eff3_model.predict(img)
      x_t = x_t.reshape(1, -1)

      # Predict the result
      result = loaded_model.predict(x_t)
      # Define class names based on your dataset
      class_names = [
          'Acnes', 'Healthy', 'Vitiligo', 'Fungal Infections',
          'Melanoma Skin Cancer and Moles', 'Eczema'
      ]
      predicted_class_index = np.argmax(result)
      predicted_class_name = class_names[predicted_class_index]
      acc = result[0][predicted_class_index]
      return acc, predicted_class_name
    except BaseException:
      return 0.0, "Invalid Image"


@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    # Check if email and password match
    if email in emails and password in passwords:
        # Redirect to the dashboard or another page
        return render_template('upload.html')
    else:
        # Handle incorrect credentials, you can render an error page or redirect to the login page
        return render_template('signin.html')
   


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = 'static/data/' + f.filename
        f.save(path)
        acc, disease = get_class(path)
        rounded_acc = round(acc * 100, 2)
     
        #yolo_results = yolo_model.predict(source=path,save=True)
        count=0
        pred=yolo_model.predict('static/data/' + f.filename, confidence=40, overlap=30).json()
        count = len(pred['predictions'])
        print(f"Number of predictions: {count}")
        yolo_results=yolo_model.predict('static/data/' + f.filename, confidence=40, overlap=30).save("prediction.jpg")
        #for r in yolo_results:
        #   for c in r.boxes.cls:
        #     count=count+1
        #source_path = 'runs/detect/predict/'+ f.filename
        source_path = "prediction.jpg"
        destination_path = 'static/images/'+ f.filename
        shutil.move(source_path, destination_path)
        #folder_path='runs/detect/predict/'
        #shutil.rmtree(folder_path)
      

        if rounded_acc < 70:
           disease = "Unable to Detect"
        K.clear_session()
    return render_template('uploaded.html',
                         title='Success',
                         predictions=disease,
                         acc=rounded_acc,
                         img_file=f.filename,total_forescore=count)
      #  disease = get_class(path)
       # K.clear_session()
    #return render_template('uploaded.html', title='Success', predictions=disease, acc=100, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)


