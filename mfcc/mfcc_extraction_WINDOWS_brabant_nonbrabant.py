# =========================================================================================================
# Extracting Mel-frequency cepstral coefficients (MFCC) from the final dataset(s)
# =========================================================================================================

number_mfcc_bands = 120

# Load various imports 
import pandas as pd
import os
import librosa
import numpy as np
import os
import pickle

# Define a function that will generate MFCC from a given audio file
def extract_features(file_name):
    # Print current audio file
    # print('Loading: "', file_name, '"')
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
features_trainingandvalidation = []
features_testing = []

# =========================================================================================================
# EXTRACT THE FEATURES FOR THE TRAINING + VALIDATION DATASET
# =========================================================================================================

# Retrieve current working directory
path = os.getcwd() 
print(path)

# In Windows this is the root path is different: We will change it to the right path
os.chdir("C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\training_validation_dataset")
path = os.getcwd() 

# Get the list of all files and directories in current working directory 
dir_list = os.listdir(path) 

for className in os.listdir(os.getcwd()):	
	print(className)
	# Navigate to the map called english (which contain all our Brabants accent samples)
	if className == "Brabants":
		for fileName in os.listdir(os.getcwd() + "\\Brabants"):
			# Generate the MFCC for the given audio file
			data = extract_features(os.getcwd() + "\\Brabants\\" + fileName)
			# Then store the resulting MFCC in the feature array under the
			# classification of the parent classifier
			features_trainingandvalidation.append([data, className])

	# Navigate to the map called korean (which contain all our Non-Brabants accent samples)
	if className == "Non-Brabants":

		for fileName in os.listdir(os.getcwd() + "\\Non-Brabants"):
			# Generate the MFCC for the given audio file
			data = extract_features(os.getcwd() + "\\Non-Brabants\\" + fileName)
			# Then store the resulting MFCC in the feature array under the
			# classification of the parent classifier
			features_trainingandvalidation.append([data, className])

# Lastly, put the resulting array in a pandas DataFrame
features_trainingandvalidation_df = pd.DataFrame(features_trainingandvalidation, columns=['feature', 'class_label'])
print('Finished feature extraction from ', len(features_trainingandvalidation_df), ' files')
print(features_trainingandvalidation_df)

# Place the pandag DataFrame in a CSV file
features_trainingandvalidation_df.to_pickle('mffc_extracted_WINDOWS_training_and_validation')

# =========================================================================================================
# EXTRACT THE FEATURES FOR THE TEST DATASET
# =========================================================================================================

# Retrieve current working directory
path = os.getcwd() 
print(path)

# In Windows this is the root path is different: We will change it to the right path
os.chdir("C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\test_dataset")
path = os.getcwd() 

# Get the list of all files and directories in current working directory 
dir_list = os.listdir(path) 

for className in os.listdir(os.getcwd()):
	
	print(className)
	# Navigate to the map called english (which contain all our Brabants accent samples)
	if className == "Brabants":
		for fileName in os.listdir(os.getcwd() + "\\Brabants"):
			# Generate the MFCC for the given audio file
			data = extract_features(os.getcwd() + "\\Brabants\\" + fileName)
			# Then store the resulting MFCC in the feature array under the
			# classification of the parent classifier
			features_testing.append([data, className])

	# Navigate to the map called korean (which contain all our Non-Brabants accent samples)
	if className == "Non-Brabants":

		for fileName in os.listdir(os.getcwd() + "\\Non-Brabants"):
			# Generate the MFCC for the given audio file
			data = extract_features(os.getcwd() + "\\Non-Brabants\\" + fileName)
			# Then store the resulting MFCC in the feature array under the
			# classification of the parent classifier
			features_testing.append([data, className])

# Lastly, put the resulting array in a pandas DataFrame
features_testing_df = pd.DataFrame(features_testing, columns=['feature', 'class_label'])
print('Finished feature extraction from ', len(features_testing_df), ' files')
print(features_testing_df)

# Place the pandag DataFrame in a CSV file
features_testing_df.to_pickle('mffc_extracted_WINDOWS_test')