'''
The code snippet below is our first model, a simple stack of 3 convolution layers 
with a ReLU activation and followed by max-pooling layers. This is very similar to 
the architectures that Yann LeCun advocated in the 1990s for image classification 
(with the exception of ReLU).
'''

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.backend import image_data_format
from keras import backend as K

from os.path import dirname, abspath
from os import walk

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


img_width, img_height = 150, 150
#img_width, img_height = 761, 800

input_shape = (150, 150, 3)
#input_shape = (3, 150, 150)


model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Conv2D(32, 3, 3, input_shape=input_shape))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3)))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

'''
On top of it we stick two fully-connected layers. We end the model with a single 
unit and a sigmoid activation, which is perfect for a binary classification. To 
go with it we will also use the binary_crossentropy loss to train our model.
'''

#basedir = dirname(abspath(__file__))
#basedir += '/data'
#basedir = '/home/ubuntu/kojak/data'
basedir = '/data/data'
#basedir = '/data/data2'
targetdir = basedir + '/train'
valdir = basedir + '/validation'

num_categories = get_num_classes(targetdir)
print('Number of categories:', num_categories)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(1))
model.add(Dense(num_categories)) # number of categories
#model.add(Activation('sigmoid'))
model.add(Activation('softmax')) # for multiclass

'''
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''

# for multiclass
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



'''
Let's prepare our data. We will use .flow_from_directory() to generate batches 
of image data (and their labels) directly from our jpgs in their respective 
folders.
'''
 
batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=180,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolders of 'data/train', and indefinitely generate
# batches of augmented image data

classes = next(walk(targetdir))[1]

train_generator = train_datagen.flow_from_directory(
        targetdir,  # this is the target directory (originally = 'data/train')
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        valdir, # (originally 'data/validation')
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes)

'''
We can now use these generators to train our model. Each epoch takes 20-30s on 
GPU and 300-400s on CPU. So it's definitely viable to run this model on CPU if 
you aren't in a hurry.
'''

#epochs=1, # original 50, 400s/epoch on cpu (20s/epoch on gpu)
'''
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
'''

nb_epoch = 25
nb_train_samples = jpg_counts(targetdir)
#nb_train_samples = 51634
nb_validation_samples = jpg_counts(valdir)

model.fit_generator(
        train_generator,
        samples_per_epoch = nb_train_samples,
        nb_epoch = nb_epoch,
        validation_data=validation_generator,
        nb_val_samples = nb_validation_samples)
model.save_weights('first_try.h5')  # always save your weights after training or during training

print('Done!')