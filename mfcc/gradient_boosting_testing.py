#===================================
#   GRADIENT BOOSTING
#===================================

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

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

# Scale the features on the X sets, this is a different scaler compared to RandomForest
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#After testing different learning rates, use the one with the highest accuracy on the validation dataset
learning_rate = 0.5
gradient_booster_classifier = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
gradient_booster_classifier.fit(X_train, y_train)
print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.2f}".format(gradient_booster_classifier.score(X_train, y_train)))
print("Accuracy score (validation): {0:.2f}".format(gradient_booster_classifier.score(X_test, y_test)))

#Evaluate the classifier by checking its accuracy and creating a confusion matrix
gradient_booster__evaluation_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gradient_booster__evaluation_classifier.fit(X_train, y_train)
predictions = gradient_booster__evaluation_classifier.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))

# save the model to disk with pickle
filename = 'gradient_boosting_final_testing.sav'
pickle.dump(gradient_booster_classifier, open(filename, 'wb'))
