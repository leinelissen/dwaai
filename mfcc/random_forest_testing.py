#===================================
#   RANDOM FOREST
#===================================

import pickle
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

features_trainingandvalidation_df = pd.read_pickle(os.path.join(os.path.curdir, 'training_validation_dataset', 'mffc_extracted_WINDOWS_training_and_validation'))
print(features_trainingandvalidation_df)

# Get dataframes from folder C:\Users\s157874\Documents\GitHub\dwaai\mfcc\example_datas. Change this to your own path. 
features_testing_df = pd.read_pickle(os.path.join(os.path.curdir, 'test_dataset', 'mffc_extracted_WINDOWS_test'))
print(features_testing_df)

# Convert features and corresponding classification labels into numpy arrays
X_train = np.array(features_trainingandvalidation_df.feature.tolist())
y_train = np.array(features_trainingandvalidation_df.class_label.tolist())

X_test = np.array(features_testing_df.feature.tolist())
y_test = np.array(features_testing_df.class_label.tolist())

# Log training and test instances
print(len(X_train), 'training instances,', len(X_test), 'testing instances')

# Scale the features on the X sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Then, setup the RandomForest classifier
classifier = RandomForestClassifier(n_estimators=100)

# Train the algorithm on the current data
classifier.fit(X_train, y_train)

# Then evaluate the predictions on the test data...
y_pred = classifier.predict(X_test)

# ...and log the output from the test evaluation
feature_imp = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
print("Classification Report")
print(classification_report(y_test, y_pred))

# save the model to disk with pickle
filename = 'random_forest_final_test.sav'
pickle.dump(classifier, open(filename, 'wb')) 