from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from PIL import Image
import numpy as np
from skimage.util import img_as_bool
from boto3.session import Session
import sys

ACCESS_KEY_ID = 'AKIARXQBP2ZVBYLBIEWV'
SECRET_KEY = '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'

session = Session(aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_KEY)

s3 = session.resource('s3')

bucket = 'image-already-upload' 

my_bucket = s3.Bucket(bucket)

brand = sys.argv[1]
model = sys.argv[2]

my_bucket.download_file(brand + "_" + model + "_" + "front",'C:/Users/HarshGupta/Desktop/Image_user/Front_view/' + brand + "_" + model + '.jpg')
my_bucket.download_file(brand + "_" + model + "_" + "rear",'C:/Users/HarshGupta/Desktop/Image_user/Rear_view/' + brand + "_" + model + '.jpg')
my_bucket.download_file(brand + "_" + model + "_" + "left",'C:/Users/HarshGupta/Desktop/Image_user/LSide_view/' + brand + "_" + model + '.jpg')
my_bucket.download_file(brand + "_" + model + "_" + "right",'C:/Users/HarshGupta/Desktop/Image_user/RSide_view/' + brand + "_" + model + '.jpg')

def orb_sim(img1, img2):
  orb = cv2.ORB_create()

  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 60]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def structural_sim(img1, img2):
  sim, diff = structural_similarity(img1, img2, full=True,  win_size=1, use_sample_covariance=False)
  return sim

data = brand + "_" + model
 
for filen in os.listdir("C:/Users/HarshGupta/Desktop/Images_org/Front_view/"):
    if(data + ".jpg" in filen):
      img = plt.imread("C:/Users/HarshGupta/Desktop/Images_org/Front_view/"+data+".jpg");

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Image_user/Front_view/"):
    if(data + ".jpg" in filen):
      imga = plt.imread("C:/Users/HarshGupta/Desktop/Image_user/Front_view/"+data+".jpg");

data_f = []
data_of=[]

orb_similarity = orb_sim(img,imga)
data_of.append(orb_similarity)

data_sf = []
img_b = np.squeeze(img)
img_c = np.squeeze(imga)
ssim = structural_sim(img_b, img_c)
data_sf.append(ssim)

maxi = data_of[0];    

maxt = data_sf[0];    

avga = (maxt+maxi)/2;

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Images_org/Rear_view/"):
    if(data + ".jpg" in filen):
      imgb = plt.imread("C:/Users/HarshGupta/Desktop/Images_org/Rear_view/"+data+".jpg");

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Image_user/Rear_view/"):
    if(data + ".jpg" in filen):
      imgc = plt.imread("C:/Users/HarshGupta/Desktop/Image_user/Rear_view/"+data+".jpg");

data_or=[]
orb_similarity = orb_sim(imgb,imgc)
data_or.append(orb_similarity)

data_sr = []
imgb = resize(imgb, (imgc.shape[0], imgc.shape[1], imgc.shape[2]), anti_aliasing=True, preserve_range=True)
img_b = np.squeeze(imgb)
img_c = np.squeeze(imgc)
ssim = structural_sim(img_b, img_c)
data_sr.append(ssim)

maxi = data_or[0];    

maxt = data_sr[0];    

avgb = (maxt+maxi)/2;

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Images_org/LSide_view/"):
    if(data + ".jpg" in filen):
      imgd = plt.imread("C:/Users/HarshGupta/Desktop/Images_org/LSide_view/"+data+".jpg");

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Image_user/LSide_view/"):
    if(data + ".jpg" in filen):
      imge = plt.imread("C:/Users/HarshGupta/Desktop/Image_user/LSide_view/"+data+".jpg");

data_ol=[]
orb_similarity = orb_sim(imgd,imge)
data_ol.append(orb_similarity)

data_sl = []
imgd = resize(imgd, (imge.shape[0], imge.shape[1], imge.shape[2]), anti_aliasing=True, preserve_range=True)
img_d = np.squeeze(imgd)
img_e = np.squeeze(imge)
ssim = structural_sim(img_d, img_e)
data_sl.append(ssim)

maxi = data_ol[0];    

maxt = data_sl[0];

avgc = (maxt+maxi)/2;

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Images_org/RSide_view/"):
    if(data + ".jpg" in filen):
      imgf = plt.imread("C:/Users/HarshGupta/Desktop/Images_org/RSide_view/"+data+".jpg");

for filen in os.listdir("C:/Users/HarshGupta/Desktop/Image_user/RSide_view/"):
    if(data + ".jpg" in filen):
      imgg = plt.imread("C:/Users/HarshGupta/Desktop/Image_user/RSide_view/"+data+".jpg");

data_ors=[]
orb_similarity = orb_sim(imgf,imgg)
data_ors.append(orb_similarity)

data_srs = []
imgf = resize(imgf, (imgg.shape[0], imgg.shape[1], imgg.shape[2]), anti_aliasing=True, preserve_range=True)
img_f = np.squeeze(imgf)
img_g = np.squeeze(imgg)
ssim = structural_sim(img_f, img_g)
data_srs.append(ssim)

maxi = data_ors[0];    

maxt = data_srs[0];

avgd = (maxt+maxi)/2;

total_score = (avga+avgb+avgc+avgd)/4;

with open('C:/Users/HarshGupta/Desktop/Images_output/out.txt', 'w') as f:
    print(total_score, file=f)