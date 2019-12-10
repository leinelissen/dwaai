#!/usr/bin/env python3

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys

# GUARD: Check if someone has supplied an argument
if len(sys.argv) < 2:
    raise Exception('No input file')

# Read an input file from the first argument on the commandline
inputFile = sys.argv[1]

# Calculate MFCC
(rate,sig) = wav.read(inputFile)
(mfcc_features, mfcc_energy) = mfcc(sig, rate, 0.025, 0.01, 13, 26, 2048)
# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate)

# Print MFCC
print(mfcc_feat)
print(mfcc_energy)