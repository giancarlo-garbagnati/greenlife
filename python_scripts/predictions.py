## Imports
from keras.preprocessing.image import load_img
from keras.models import load_model

import numpy as np
from scipy.misc import imresize, imread
from os import listdir

import json
from pprint import pprint

## Helper fxns

def scale_images(dirpath, newsize = (299,299)):
    imagelist = [dirpath + '/' + f for f in listdir(dirpath) if '.jpg' in f[-4:].lower()]
    
    imgs = [imresize(imread(img), newsize).astype('float32')[:,:,:3] 
            for img in imagelist]
    # the [:,:,:3] is to prevent alpha channels from ruining the party
    
    print(np.array(imgs).shape)
    
    
    return (np.array(imgs) / 255) , imagelist

def json_to_classes_list(dirpath, verbose=False):
    with open(dirpath) as json_data:
        classes = json.load(json_data)

    classes = {int(x):y for x,y in classes.items()}

    if verbose:
        pprint(classes)
        
    return classes

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




## LOADING MODEL + MODEL ARCHITECTURE

modelloc = '~/model/model.h5'

model = load_model(modelloc)

## IMAGE PROCESSING

#input_shape = (299, 299, 3)
img_width, img_height = 299, 299
classes = json_to_classes_list('classeslist.json')
preddir = 'pred'
X, imglist = scale_images(preddir)

## PREDICTIONS

preds = model.predict(X)
for i, pred in enumerate(preds):
    top3 = pred.argsort()[::-1][:3]
    print('Top prediction/s for', imglist[i], ':')
    print('\t(1)', classes[top3[0]], 'at', pred[top3[0]])
    print('\t(2)', classes[top3[1]], 'at', pred[top3[1]])
    print('\t(3)', classes[top3[2]], 'at', pred[top3[2]])
    print()

