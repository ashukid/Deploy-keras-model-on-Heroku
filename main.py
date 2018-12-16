from keras.models import model_from_json
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.optimizers import Adam
import pickle
import numpy as np
import base64
from flask import Flask,jsonify, render_template, request
from PIL import Image
import io
import cv2
from urllib.parse import unquote

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def toRGB(image):
    return cv2.cvtColor(np.array(image,dtype='float32'), cv2.COLOR_BGR2RGB)

def getFlowerClass(image):

    image=stringToImage(image)
    image=toRGB(image)
    im = preprocess_input(image)
    im = np.expand_dims(im,axis=0)

    json_file = open('./model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("./model.h5")
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    with open('./102labels_map.pickle','rb') as file:
        label_map = pickle.load(file)
    

    print('predicting')
    prediction = model.predict(im)[0]
    prediction = prediction.argsort()[::-1]
    pred = prediction[0]
    for x,p in label_map.items():
        if(p == pred):
            return x 


app=Flask(__name__)
@app.route('/api',methods=['POST'])
def photoRecognize():

    im=request.form['image']
    im=unquote(im)
    answer=getFlowerClass(im)
    return jsonify(status='OK', results=answer)

# @app.route('/')
# def main():
#     return render_template('index.html')

if __name__=='__main__':
    app.run()
