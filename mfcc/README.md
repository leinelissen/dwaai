# Python Pipeline
The Python pipeline is setup to train models on sample data. First, it converts input samples into MFCCs, on which a number of different models can be trained. Lastly it contains a script which can be used to input a sample for classification for all models. The current models are supported:
* 1D Convolutional Network (using Keras)
* Gradient Boosting
* Random Forest

## Installing
The following packages are required for operation: 
* `pandas`
* `librosa`
* `tensorflow`
* `numpy`
* `sklearn`
* `keras`

Install them all as follows
```
pip install pandas librosa tensorflow numpy sklearn keras
```
-- OR --
```
pip3 install pandas librosa tensorflow numpy sklearn keras
```

## Running Code
Due to some initial difficulties with traversing paths in different OSes, some files - notable MFCC extraction - have differing versions for Windows and UNIX-bases OSes. Use the files that matches your OS.

### MFCC Extraction
The `mfcc_extraction_WINDOWS_brabant_non_brabant.py` script will fetch all files from the test and training dataset and convert them into MFCCs. These datapoints will then be saved with pickle for further processing later.
```
python mfcc_extraction_WINDOWS_brabant_nonbrabant.py
```

### Model training
All models are trained with their respective files, which in turn read the pickle file generated in the previous step, and then train their models with the supplied datapoints. All outputted models are saved to disk, so that they can be loaded when samples need to be analysed. 
```
python 1D-Convolutional Network.py
python gradient_boosting.py
python random_forest.py
```

### Usage in result pipeline
As soon as all models are trained, samples can be checked against these models using the `deployment_code.py` script. It accepts an absolute path as an argument and outputs the result for the Brabantness classification back to the console as such:
```
python deployment_code.py /fully/qualified/path/to/audio/file.wav
0.58,0.7190020281775362,0.98751056
```
The output is coded as follows: `gradient-boosting,random-forest,convolutional-neural-network`. The script is automatically called by the NodeJS back-end when using the included web-application.
