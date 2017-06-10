from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, RMSprop

# gets jpg count for all images in given directory and all subdirectories
def jpg_counts(dirpath, verbose=False):
    from os import listdir, walk
    #from os.path import isfile, join
    
    # list of all subdirectories
    dirlist = [x[0] for x in walk(dirpath)][1:]
    
    # list of all images in this directory
    imagelist = [f for f in listdir(dirpath) if '.jpg' in f[-4:].lower()]
    if verbose:
        print(len(imagelist),'\n')
    
    # get all images in all subdirectories
    #print(dirlist)
    for currdir in dirlist:
        allfiles = [f for f in listdir(currdir)]
        imagelistsubdir = [f for f in listdir(currdir) if '.jpg' in f[-4:].lower()]
        imagelist += imagelistsubdir
        if verbose:
            if len(allfiles) != len(imagelistsubdir):
                print(currdir, len(imagelistsubdir), 'out of', len(allfiles), 'EXTRA NON JPG FILES')
            else:
                print(currdir, len(imagelistsubdir), 'out of', len(allfiles))

    if verbose:
        print(len(imagelist))
    return len(imagelist)

def get_num_classes(dirpath, verbose=False):
    from os import walk
    
    # get list of all direct subdirectories
    dirlist = next(walk(dirpath))[1]
    
    if verbose:
        print('Classes found:')
        for d in dirlist:
            print(d)
    
    return len(dirlist)


basedir = '/Users/ggarbagnati/ds/metis/metisgh/sf17_ds5/local/Projects/05-Kojak'
targetdir = basedir + '/data/train'
valdir = basedir + '/data/validation'
'''
basedir = '/data/data'
targetdir = basedir + '/train'
valdir = basedir + '/validation'
'''

#img_width, img_height = 761, 800
img_width, img_height = 299, 299 # inception likes 299x299
nb_train_samples = jpg_counts(targetdir)
nb_validation_samples = jpg_counts(valdir)
nb_categories = get_num_classes(targetdir)
batch_size = 32
nb_epoch = 1

# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False)
model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
#x = base_model.output
x = model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
#x = Dense(1024, activation='relu', name='fc_1')(x)
x = Dense(1024, activation='relu', name='fc_1')(x) # num of neurons in the layer
predictions = Dense(nb_categories, activation='softmax')(x)

'''
# Freeze convolutional layers
for layer in model.layers:
    layer.trainable = False
'''
"""

fullmodel = Model(input=model.input, output=predictions)


# Freeze convolutional layers
for layer in model.layers:
    layer.trainable = False


fullmodel.compile(optimizer=RMSprop(lr = .00001), loss = 'categorical_crossentropy', metrics=['accuracy'])
#fullmodel.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])

"""

fullmodel = load_model('leafincepmodel-ft-3.h5')

train_datagen = ImageDataGenerator(rotation_range=180,
                                    rescale = 1./255.,
                                    shear_range = .2,
                                    zoom_range = .2,
                                    horizontal_flip = True)



# Inception has a custom image preprocess function
test_datagen = image.ImageDataGenerator(rescale=1./255)

generator_train = train_datagen.flow_from_directory(
        targetdir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

generator_test = test_datagen.flow_from_directory(
        valdir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

fullmodel.fit_generator(generator_train,
            samples_per_epoch = nb_train_samples,
            nb_epoch = nb_epoch,
            validation_data = generator_test,
            nb_val_samples = nb_validation_samples)



# if I get time...

#start fine-tuning
# unfreeze the top 2 inception blocks
for layer in fullmodel.layers[:172]:
   layer.trainable = False
for layer in fullmodel.layers[172:]:
   layer.trainable = True

# use SGD with a low learning rate
fullmodel.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy'])

# Train the top 2 inception blocks
fullmodel.fit_generator(generator_train,
                        samples_per_epoch = nb_train_samples,
                        nb_epoch = nb_epoch,
                        validation_data = generator_test,
                        nb_val_samples = nb_validation_samples)


fullmodel.save('leafincepmodel-test.h5')

'''
model_json = fullmodel.to_json()
with open('incep_3_multi.json', 'w') as json_file:
    json_file.write(model_json)
fullmodel.save_weights('incep_3_multi.h5')
'''

print('Done!')
