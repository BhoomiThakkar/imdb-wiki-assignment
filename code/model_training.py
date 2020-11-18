import ssl
import numpy as np
import pandas as pd
from pickle import dump
from sklearn import metrics
from keras.models import Model
from utility import model_training_functions
from keras.applications.vgg16 import VGG16

'''
1. Feature Extract from VGG-16 for all images in the dataset
2. Save these features
3. Divide data into training, cross-validation, test set
4. Train the age classification model
5. Print model performance metrics 
'''

# To avoid SSL error while downloading VGG-16 model
ssl._create_default_https_context = ssl._create_unverified_context

# Variable and data declarations
nclasses=101  # no of age classes
nepochs=20  # no of training epochs
trainProp=0.75 # prop of training data

# Directory configurations
project_dir='/Users/bhoomithakkar/PycharmProjects/chicago_trial/'
image_directory=project_dir+'wiki/'
feature_directory=project_dir+'model/'
metadata_directory=project_dir+'data/'

metadata=pd.read_csv(metadata_directory+'wiki.csv')

# Use the pretrained weight matrix
vggmodel = VGG16()
vggmodel = Model(inputs=vggmodel.inputs, outputs=vggmodel.layers[-2].output)
print(vggmodel.summary())

# Extract features for all images (pickle used for faster serialization/de-serialization)
# we can use JSON/XML/txt files too instead
features = model_training_functions.extract_features(image_directory)
print('Extracted Features: ', len(features))
dump(features, open(feature_directory+'features.pkl', 'wb'))

# Create a dataframe for the features extracted (to merge with age)
features_df=pd.DataFrame()
features_df['filepath_index'] = list(features.keys())
features_df['image_features'] = list(features.values())
features_df_1=pd.merge(metadata, features_df, on=['filepath_index'])


# Use image features as input
X=[features_df['image_features'][i][0] for i in range(len(features_df_1))]
# Use corresponding age values as output
y=features_df_1['age'].tolist()

# Proportion of observations used for training
n_observations=int(len(features_df_1)*trainProp)

# Re-shaping the training and test vectors
Xtrain=X[:n_observations]
Xtrain=np.asarray(Xtrain)
Xtrain=Xtrain.reshape(len(Xtrain), 4096).astype("float32")
ytrain=y[:n_observations]
ytrain=np.asarray(ytrain)
ytrain=ytrain.reshape(len(ytrain), 1).astype("float32")

Xtest=X[n_observations:]
Xtest=np.asarray(Xtest)
Xtest=Xtest.reshape(len(Xtest), 4096).astype("float32")
ytest=y[n_observations:]
ytest=np.asarray(ytest)
ytest=ytest.reshape(len(ytest), 1).astype("float32")


# Training the Age Classification model
ageclassification_model=model_training_functions.define_model(nclasses)
ageclassification_model.fit([Xtrain], ytrain, epochs=nepochs, verbose=2)
ypred=ageclassification_model.predict(Xtest)
ypredlabels=[np.argmax(i) for i in ypred]
ytestlabels=[int(i[0]) for i in ytest.tolist()]

# Print Precision, Recall, F-1
print(metrics.classification_report(ytestlabels, ypredlabels))

