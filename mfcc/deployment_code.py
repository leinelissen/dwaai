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

# Prevent tensorflow from printing stuff
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
sys.stderr = stderr

# Guard: check if there is a argument given, otherwise raise error
if len(sys.argv) < 2:
    raise Exception('No input file')

# Get the directory of the file, so that we can reference to the stored models
modelPath = os.path.dirname(os.path.abspath(__file__))

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
# 2. Gradient Boosting
# 3. One-Dimensional Convolutional Network
# =========================================================================================================

# 1. Load Random Forest Classifier
filename = os.path.join(modelPath, 'random_forest_final.sav')
loaded_model = pickle.load(open(filename, 'rb'))
predictions_rf = loaded_model.predict_proba(Xnew)

# 2. Load Gradient Boosting
filename = os.path.join(modelPath, 'gradient_boosting_final.sav')
loaded_model = pickle.load(open(filename, 'rb'))
predictions_gb = loaded_model.predict_proba(Xnew)

# 3. Load One-Dimensional Convolutional Network and make prediction
K.clear_session()
model = tf.keras.models.load_model(os.path.join(modelPath, '1D_Conv_model.model'), compile=False)
# ---------------------------------------------------------------------------------------------------------
# reshape
xtss = Xnew.shape
XKeras = np.reshape(Xnew.values, (xtss[0], xtss[1], 1))
# Make a prediction 
predictions_cnn = model.predict_proba(XKeras)
# print("X=%s, Predicted=%s" % (XKeras, predictions_cnn))

# Output all predictions
print(str(predictions_rf[0][0]) + ',' + str(predictions_gb[0][0]) + ',' + str(predictions_cnn[0][0]))
sys.stdout.flush()
