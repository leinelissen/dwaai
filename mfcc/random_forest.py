#===================================
#   RANDOM FOREST CLASSIFIER FOR DWAAI
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

# Get dataframe from folder C:\Users\s157874\Documents\GitHub\dwaai\mfcc\example_datas
featuresdf = pd.read_pickle('C:\\Users\\s157874\\Documents\\GitHub\\dwaai\\mfcc\\training_validation_dataset\\mffc_extracted_WINDOWS_training_and_validation') 
print(featuresdf)

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Split our dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Log training and test instances
print(len(X_train), 'training instances,', len(X_test), 'testing instances')

# Scale the features on the X sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Then, setup the RandomForest classifier. 
# n_estimators are the number of trees in the classifier. We set this to the value of 10, since our dataset is relatively small
# max_features are the number of features to consider when looking for the best split. We set this at 120, since we have 120 features extracted out of the mfcc's. 
# random_state controls the randomness. We set this at at the value of 10 aswell, since our dataset is relatively small   
classifier = RandomForestClassifier(n_estimators=10, random_state=10, max_features=120)  

# Train the algorithm on the current data
classifier.fit(X_train, y_train)

# Then evaluate the predictions on the test data...
y_pred = classifier.predict(X_test)

# ...and log the output from the test evaluation
feature_imp = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
print("Classification Report")
print(classification_report(y_test, y_pred))

# save the model to disk with pickle
filename = 'random_forest_final.sav'
pickle.dump(classifier, open(filename, 'wb'))