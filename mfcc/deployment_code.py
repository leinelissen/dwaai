import sys
# =========================================================================================================
# DEPLOYMENT CODE! FISSA
# First we import our dependencies
# =========================================================================================================
number_mfcc_bands = 120

# Load various imports 
import pandas as pd
import os
import librosa
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# Guard: check if there is a argument given, otherwise raise error
if len(sys.argv) < 2:
    raise Exception('No input file')

# Get argument from node.js. This is the absolute path from the generated audio file that is saved in "..."
inputFile_path = sys.argv[1]

# =========================================================================================================
# Preprocessing: Extracting Mel-frequency cepstral coefficients (MFCC) 
# =========================================================================================================

# Define a function that will generate MFCC from a given audio file
def extract_features(file_name):
    try:
        # Load audio file into librosa
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # Generate MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=number_mfcc_bands)
        mfccsscaled = np.mean(mfccs.T,axis=0)  
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return None 
    # Return the MFCC
    return mfccsscaled
# Create an empty array with the features that are about to be generated
features_deploy = []

# Extract features
incoming_data = extract_features(inputFile_path)
# Then store the resulting MFCC in the feature array 
features_deploy.append([incoming_data])
# Put features in a pandas dataframe
Xnew = pd.DataFrame(features_deploy)
print(Xnew, 'After pandas')
# reshape
xtss = Xnew.shape
print(incoming_data)
print(Xnew)
print(xtss)
print(xtss[0])
Xnew = np.reshape(Xnew, (xtss[0], xtss[1], 1))
print(Xnew)
sys.stdout.flush()

# =========================================================================================================
# Load in your Machine Learning/Deep Learning Model to make prediction 
# 1. One-Dimensional Convolutional Network
# 2. Random Forest
# 3. Gradient Boosting
# =========================================================================================================

# 1. Load One-Dimensional Convolutional Network and make prediction
import tensorflow as tf
from keras.models import load_model
model = tf.keras.models.load_model('C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\1D_Conv_model.model')
# ---------------------------------------------------------------------------------------------------------
# Generate 2d classification dataset
CATEGORIES = ["Brabants", "Non-Brabants"]
# Make a prediction 
prediction = model.predict_proba(Xnew)
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
# for i in range(len(Xnew)):
	#print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# 2. Load Random Forest Classifier
filename = 'C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\random_forest_final.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = classifier.predict_proba(Xnew)


# 3. Load Gradient Boosting
# filename = 'C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\gradient_boosting_final.sav'
# loaded_model = pickle.load(open(filename, 'rb'))

# =========================================================================================================
# Make a prediction for the outcome of the following models
# 1. One-Dimensional Convolutional Network
# 2. Random Forest
# 3. Gradient Boosting
# =========================================================================================================

# 1. Make prediction for One-Dimensional Convolutional Network

# 2. Make prediction for Random Forest

# =========================================================================================================
# Convert to % and send to Node.js
# =========================================================================================================

