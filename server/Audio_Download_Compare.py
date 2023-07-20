import sys
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.patheffects as path_effects
import acoustid
import chromaprint
import seaborn as sns
from fuzzywuzzy import fuzz
from boto3.session import Session

ACCESS_KEY_ID = 'AKIARXQBP2ZVIIEGC2VH'
SECRET_KEY = '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'

session = Session(aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_KEY)

s3 = session.resource('s3')

bucket = 'audio-already-upload'

my_bucket = s3.Bucket(bucket)

brand = sys.argv[1]
model = sys.argv[2]

my_bucket.download_file(brand + "_" + model + "_" + "accelarate",'C:/Users/HarshGupta/Desktop/Audio_user/Accelerating/' + brand + "_" + model + '.wav')
my_bucket.download_file(brand + "_" + model + "_" + "deaccelarate",'C:/Users/HarshGupta/Desktop/Audio_user/Deaccelerating/' + brand + "_" + model + '.wav')

data = brand + "_" + model
 
for filen in os.listdir("C:/Users/HarshGupta/Desktop/Audio_org/Accelerating/"):
    if(data + ".wav" in filen):
      file1 = "C:/Users/HarshGupta/Desktop/Audio_org/Accelerating/"+ data +".wav"

file2 = "C:/Users/HarshGupta/Desktop/Audio_user/Accelerating/" + data + ".wav"

signal,sr=librosa.load(file1, sr=22050)

duration, fp_encoded = acoustid.fingerprint_file(file1)
fingerprint1 = chromaprint.decode_fingerprint(fp_encoded)
duration, fp_encoded2 = acoustid.fingerprint_file(file2)
fingerprint2 = chromaprint.decode_fingerprint(fp_encoded2)

similarity = fuzz.ratio(fingerprint1, fingerprint2)

total_score = similarity

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Audio_org/Deaccelerating/"):
    if(data + ".wav" in filen):
      file3 = "C:/Users/HarshGupta/Desktop/Audio_org/Deaccelerating/"+data+".wav"

file4 = "C:/Users/HarshGupta/Desktop/Audio_user/Deaccelerating/" + data + ".wav"

duration, fp_encoded3 = acoustid.fingerprint_file(file3)
fingerprint3, version = chromaprint.decode_fingerprint(fp_encoded3)
duration, fp_encoded4  = acoustid.fingerprint_file(file4)
fingerprint4, version = chromaprint.decode_fingerprint(fp_encoded4)

similarity2 = fuzz.ratio(fingerprint3, fingerprint4)

total_score2 = similarity2;

final_score = (similarity+similarity2)/2;

with open('C:/Users/HarshGupta/Desktop/Audio_output/Final_out.txt', 'w') as f:
    print(final_score, file=f)