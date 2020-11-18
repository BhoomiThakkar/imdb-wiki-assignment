from os import listdir
from pickle import load
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input


'''
Function definitions for 
- Feature extraction from VGG
- Feature loading from pkl file
- Defining the Age Classification model 
'''


# Extract features from each photo from the sub-directories
def extract_features(model, directory):

    # extract features from each photo
    features = dict()

    for subdir in listdir(directory):
        for name in listdir(directory + subdir):
            filename = directory + subdir + '/' + name
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            dir_type=directory.split('/')[-2]
            features[dir_type + '/' + subdir + '/' + name] = feature
            # print(name)

    return features


# Load photo features
def load_photo_features(filename,keys):

    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in keys}

    return features


# Define the age classification model architecture
def define_model(n_classes):

    input_layer = Input(shape=(4096,))
    dropout_layer = Dropout(0.5)(input_layer) # Dropout layer
    fully_connected1 = Dense(256, activation='relu')(dropout_layer)
    fully_connected2 = Dense(256, activation='relu')(fully_connected1)
    output_layer = Dense(n_classes, activation='softmax')(fully_connected2)

    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model

