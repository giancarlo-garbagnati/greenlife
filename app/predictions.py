## Imports
from __future__ import print_function
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from keras.models import load_model
from scipy.misc import imresize, imread
from pprint import pprint
import sys
import os
import json
import numpy as np

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def initialize_model(): 
    ## LOADING MODEL + MODEL ARCHITECTURE
    modelloc = './model/model.h5'
    global model 
    model = load_model(modelloc)

model = None;
initialize_model()

def json_to_classes_list(dirpath, verbose=False):
    with open(dirpath) as json_data:
        classes = json.load(json_data)
    classes = {int(x):y for x,y in classes.items()}
    if verbose:
        pprint(classes)
    return classes

classes = json_to_classes_list('classeslist.json')

## Helper fxns

def scale_images(dirpath, newsize = (299,299)):
    imagelist = [dirpath + '/' + f for f in listdir(dirpath) if '.jpg' in f[-4:].lower()]
    imgs = [imresize(imread(img), newsize).astype('float32')[:,:,:3] 
            for img in imagelist]
    # the [:,:,:3] is to prevent alpha channels from ruining the party
    return (np.array(imgs) / 255) , imagelist

def scale_image(imgdir, newsize =(299,299)):
    img = imresize(imread(imgdir), newsize).astype('float32')[:,:,:3] 
    # the [:,:,:3] is to prevent alpha channels from ruining the party
    return np.array([img]) / 255


def parse_plant_name(plantstring, replace_underscores=None, remove_parentheses=False):
    ## basic parsing
    underscore_i = plantstring.find('_')
    plantsp = plantstring[:underscore_i]
    healthstatus = plantstring[underscore_i+1:]
    
    ## special cases
    # special case for bell peppers (2 cases)
    if plantsp == 'Pepper,':
        plantsp = 'bell_pepper'
        underscore_i = healthstatus.find('_')
        healthstatus = healthstatus[underscore_i+1:]
    
    # special case for 'tomato_Spider_mites??Two-spotted_spider_mite' (1 case)
    if healthstatus == 'Spider_mites??Two-spotted_spider_mite':
        healthstatus = 'Spider_mites_(Two-spotted_spider_mite)'
    
    ## extra parameters
    if replace_underscores != None:
        plantsp = plantsp.replace('_', replace_underscores)
        healthstatus = healthstatus.replace('_', replace_underscores)
        
    if remove_parentheses:
        plantsp = plantsp.replace('(','')
        plantsp = plantsp.replace(')','')
        healthstatus = healthstatus.replace('(','')
        healthstatus = healthstatus.replace(')','')
    
    return plantsp, healthstatus

def process_images():
    ## IMAGE PROCESSING
    #input_shape = (299, 299, 3)
    img_width, img_height = 299, 299
    classes = json_to_classes_list('classeslist.json')
    preddir = '~/images'
    X, imglist = scale_images(preddir)
    return X, imglist

def process_image(img_dir):
    ## IMAGE PROCESSING
    #input_shape = (299, 299, 3)
    X = scale_image(img_dir)
    return X


def predict_multi(X, imglist):
    ## PREDICTIONS
    preds = model.predict(X)
    for i, pred in enumerate(preds):
        top3 = pred.argsort()[::-1][:3]
        print('Top prediction/s for', imglist[i], ':')
        print('\t(1)', classes[top3[0]], 'at', pred[top3[0]])
        print('\t(2)', classes[top3[1]], 'at', pred[top3[1]])
        print('\t(3)', classes[top3[2]], 'at', pred[top3[2]])
        print()
    
def predict_single(X, imgdir):
    ## PREDICTIONS
    preds = model.predict(X)
    top3 = preds[0].argsort()[::-1][:3]
    return top3, preds[0]

def format_results(top3, pred):
    results = []
    for i in range(3):
        results.append({ 'class': str(classes[top3[i]]), 'confidence': str(pred[top3[i]]) }) 
    return results

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files, file=sys.stderr)
        if 'predict_image' not in request.files:
            print('No file part', file=sys.stderr)
            return jsonify({'success': 'false', 'error':'NO_FILE_UPLOADED'})
        file = request.files['predict_image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file', file=sys.stderr)
            return jsonify({'success': 'false', 'error':'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            X = process_image(file)
            top3, pred = predict_single(X, filename)
            return jsonify({'success': 'true', 'results': format_results(top3, pred)})
        else:
            return jsonify({'success': 'false', 'error':'INVALID_FILE'})

#def post_image(img):
    #save image
    #process image
    #predict
    #return response
    
