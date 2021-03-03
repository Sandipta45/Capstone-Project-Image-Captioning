# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:13:25 2020
@author: admin
"""
import streamlit as st
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import pandas as pd
import h5py
import pathlib
import boto3

#import tensorflow as tf
import tensorflow as tf
import keras

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm.notebook import tqdm as tqdm
tqdm().pandas()
#model = load_model('C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5')
#dataset_text = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text"
#filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#load the data 
#filename = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text/Flickr8k.token.txt"
filename = "Flickr8k.token.txt"
# Loading a text file into memory

def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#filename = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text/Flickr_8k.trainImages.txt"
#filename = "Flickr_8k.trainImages.txt"
#def load_photos(filename):
#    file = load_doc(filename)
#    photos = file.split("\n")[:-1]
#    return photos
#train_imgs = load_photos(filename)

#train_imgs
#from zipfile import ZipFile 
#file_name = "Flickr8k_images.zip" 
#with ZipFile(file_name, 'r') as zip:
#    img_path=zip.read() 
#img_path = open('Flickr8k_images.zip')
#img_path = "https://raw.githubusercontent.com/gulfairus/project2/tree/master/Flicker8k_Dataset"
#img_path = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_Dataset/Flicker8k_Dataset"

#img_path = "Flicker8k_Dataset"
#train_imgs=[]
#for img in tqdm(os.listdir(img_path)):
#    train_imgs.append(img)

#import plotly.express as px
#def dict_pics(train_imgs):
#    pics={}
#    for i in range(len(train_imgs)):
#        #    pics[] = {train_imgs[i]: '/'.join([img_path, train_imgs[i]])}
#        pics[train_imgs[i]] = '/'.join([img_path, train_imgs[i]])
#    return pics
#pics = dict_pics(train_imgs)

#pic = st.selectbox("Picture choices", list(pics.keys()), 0)

    #import matplotlib.pyplot as plt
    #import argparse

def extract_features(filename, model_2, shape, preprocess_input):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize(shape)
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    #image = image/127.5
    #image = image - 1.0
    feature = model_2.predict(image)
    return feature
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
            return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startcap'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endcap':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption
#
max_length = 34
tokenizer = load(open("tokenizer_project1.p","rb"))
model_2_xception = Xception(include_top=False, pooling="avg")
#model_1_inceptionv3 = InceptionV3( weights = 'imagenet' )
#model_2_inceptionv3 = Model(model_1_inceptionv3.input, model_1_inceptionv3.layers[-2].output)
#model_2_mobilenet = MobileNet(weights = 'imagenet', include_top=False, pooling='avg')
#model_1_resnet50 = ResNet50( weights='imagenet', input_shape=(224,224,3))
#model_2_resnet50 = Model(model_1_resnet50.input, model_1_resnet50.layers[-2].output)
preprocess_input_xception = keras.applications.xception.preprocess_input
#preprocess_input_inceptionv3 = keras.applications.inception_v3.preprocess_input
#preprocess_input_resnet50 = keras.applications.resnet50.preprocess_input
#preprocess_input_mobilenet = keras.applications.mobilenet.preprocess_input
shape_xception = (299,299)
#shape_inceptionv3 = (299,299)
#shape_resnet50 = (224,224)
#shape_mobilenet = (224,224)


#model = load_model('C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5')

client = boto3.client('s3', 
                      aws_access_key_id='',
                      aws_secret_access_key='')
client.download_file('project1xception',
                     'model_9.h5',
                     'model_9.h5')
model_xception = load_model('model_9.h5')

#client = boto3.client('s3', 
#                      aws_access_key_id='AKIAJIONF2XSYTTKQFDQ',
#                      aws_secret_access_key='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')
#client.download_file('project1resnet50',
#                     'model_9.h5',
#                     'model_9.h5')
#model_resnet50 = load_model('model_9.h5')
#client3 = boto3.client('s3', 
#                      aws_access_key_id='AKIAJIONF2XSYTTKQFDQ',
#                      aws_secret_access_key='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')
#client.download_file('project1mobilenet',
#                     'model_9.h5',
#                     'model_9.h5')

#model_mobilenet = load_model('model_9.h5')
#client4 = boto3.client('s3', 
#                      aws_access_key_id='AKIAJIONF2XSYTTKQFDQ',
#                      aws_secret_access_key='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')
#client.download_file('project1inceptionv3',
#                     'model_9.h5',
#                     'model_9.h5')
#model_inceptionv3 = load_model('model_9.h5')

#model.save('/models/model_9.h5')

#xception_model = Xception(include_top=False, pooling="avg")


st.write("""
# Image Captioning Prediction
""")
st.write("This is a simple image captioning web app to predict caption of the image")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is not None:
    photo_xception = extract_features(file, model_2_xception, shape_xception, preprocess_input_xception)
    #photo_inceptionv3 = extract_features(file, model_2_inceptionv3, shape_inceptionv3, preprocess_input_inceptionv3)
    #photo_mobilenet = extract_features(file, model_2_mobilenet, shape_mobilenet, preprocess_input_mobilenet)
    #photo_resnet50 = extract_features(file, model_2_resnet50, shape_resnet50, preprocess_input_resnet50)
#photo = extract_features(pics[pic], xception_model)
#    img = Image.open(pics[pic])
    description_xception = generate_desc(model_xception, tokenizer, photo_xception, max_length)
    #description_inceptionv3 = generate_desc(model_inceptionv3, tokenizer, photo_inceptionv3, max_length)
    #description_mobilenet = generate_desc(model_mobilenet, tokenizer, photo_mobilenet, max_length)
    #description_resnet50 = generate_desc(model_resnet50, tokenizer, photo_resnet50, max_length)
    #desc_table=pd.DataFrame()
    #desc_table['kerasapp']=['xception','inceptionV3','mobilenet','resnet50']
    #desc_table['caption']=[description_xception,description_inceptionv3,description_mobilenet,description_resnet50]
#st.image(pics[pic], use_column_width=True, caption=0)
#st.image(pics[pic], use_column_width=True, caption=description)
    #st.image(file, use_column_width=True, caption=st.write(desc_table))
    st.image(file, use_column_width=True, caption=description_xception)