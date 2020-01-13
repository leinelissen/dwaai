# =========================================================================================================
# DEPLOYMENT CODE! FISSA
# First we import our dependencies
# =========================================================================================================
number_mfcc_bands = 120

# Load various imports 
import sys
import pandas as pd
import os
import librosa
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

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
features_deploy.append(incoming_data)
# Put features in a pandas dataframe
Xnew = pd.DataFrame(features_deploy)
sys.stdout.flush()

# =========================================================================================================
# Load in your Machine Learning/Deep Learning Model to make prediction 
# 1. Random Forest
# 2. Gradient Boostin
# 3. One-Dimensional Convolutional Network
# =========================================================================================================

# 1. Load Random Forest Classifier
filename = os.path.join(os.path.curdir, 'random_forest_final.sav')
loaded_model = pickle.load(open(filename, 'rb'))
predictions_rf = loaded_model.predict_proba(Xnew)

# 2. Load Gradient Boosting
filename = os.path.join(os.path.curdir, 'gradient_boosting_final.sav')
loaded_model = pickle.load(open(filename, 'rb'))
predictions_gb = loaded_model.predict_proba(Xnew)

# Print both predictions
print(str(predictions_rf[0][0]) + ',' + str(predictions_gb[0][0]))

# 3. Load One-Dimensional Convolutional Network and make prediction
K.clear_session()
model = tf.keras.models.load_model(os.path.join(os.path.curdir, '1D_Conv_model.model'), compile=False)
# ---------------------------------------------------------------------------------------------------------
# reshape
xtss = Xnew.shape
XKeras = np.reshape(Xnew, (xtss[0], xtss[1], 1))
print(XKeras)
# Make a prediction 
prediction = model.predict_proba(XKeras)
print(prediction)