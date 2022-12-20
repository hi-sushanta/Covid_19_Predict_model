import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.preprocessing.image import load_img

import tensorflow as tf

CLASS_NAMES = ['COVID', 'NORMAL', 'Viral Pneumonia']
st.title("Covid 19 CT-SCAN Detection")
image = Image.open(r"C:\Users\hiwhy\OneDrive\Documents\Github_project\covid_19_ct_scan_model\covid_image.jpg")
st.image(image, width=500)

st.header("Upload the CT-SCAN Image")
ct_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

model_path = r"C:\Users\hiwhy\OneDrive\Documents\Github_project\covid_19_ct_scan_model\covid_detect"
model_load = tf.keras.models.load_model(model_path)

def predict_covid(image ,model,img_size=244):

    """
      Imports an image located at filename, makes a prediction on it with
      a trained model and plots the image with the predicted class as the title.
    """
    image = load_img(image,target_size=(img_size,img_size))
    pred = model.predict(tf.expand_dims(image, axis=0))
    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = CLASS_NAMES[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = CLASS_NAMES[int(tf.round(pred)[0][0])]  # if only one output, round

    return pred_class


def draw_image(frame, pesent_condition):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    frame = cv2.putText(frame,pesent_condition,(int(frame_w/9) , int(frame_h/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)
    return frame

if ct_image is not None:
    # Read the file and convert it to opencv image
    raw_bytes = np.asarray(bytearray(ct_image.read()), dtype=np.uint8)
    # load image in BGR channel order
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    col1.image(ct_image)
    col1.text("Original Image")
    presen_condition = predict_covid(ct_image,model_load)
    print(presen_condition)
    pred_image = draw_image(image_bgr,presen_condition)
    col2.image(pred_image[:,:,::-1])