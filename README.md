# TVS_Epic_IT_Challenge

This project was made in collaboration with @HarshGupta02 (https://github.com/HarshGupta02) for tvs_it_challenge hackathon

In this we were suppsed to automate the entire evaluation process of used automobiles such that no human intervention would be required in between.
For that we devided our project into client side and server side.

## Installation

Since ml part is done on colab most of libraries will be pre-installed, here are few which still needs to be included  

```js
1. pip install pyacoustid
```
```js
2. pip install fuzzywuzzy[speedup]
```
```js
3. pip install typing
```
```js
4. pip install --upgrade category_encoders
```
```js
5. pip install xgboost
```


## Client Side

This is basically the frontend of our website where client/user will enter all the information regarding his vehicle like Brand,Year,Model_name,Ownershipn and Location and then to make the evaluation even better we asked users to enter their vehicle's accelerating and deaccelerating sounds and its 4 sided view for audio and image comparison as well

![WhatsApp Image 2023-07-24 at 17 38 35](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/f4661e72-fc08-4132-b09b-62573a869aac)

## Server Side

This is where all the comparisons are done in the backendand and score generated will be shown in the fronted. It heppens in few stpes:

### Image Comparison
For image comparison, images provided by the user will be located in the database by their brand_model name and 4-sided view comparisomn will be done with the original(new) pics of that brand_model using orb and structural similarity. Score for each view comparison will be generated and avg of them is taken.

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/81a30d51-3c8c-4b52-87e6-e0f18e8c3b5b)


This portion shows the orb and structural similarity functions used later in code for genertingscore based comparison

```js
def orb_sim(img1, img2):
  orb = cv2.ORB_create()
  
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #to match keypoint descriptor finds the best match for each descriptor from a query image
    
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 60]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def structural_sim(img1, img2):
  sim, diff = structural_similarity(img1, img2, full=True,  win_size=1, use_sample_covariance=False) #structural_similarity(im1, im2, *, win_size=None, gradient=False, data_range=None,
  return sim #The side-length of the sliding window used in comparison
```

#### Below snippets shows the comparison of front view and similar is repeated for all 3 remaining side views

Here same image file is being searched in datebase using brand_model (here file is that speific brand_model name)

```js
for filen in os.listdir("drive/MyDrive/Image_org/Front_view/"):
    if(data + ".jpg" in filen):
      img = plt.imread("drive/MyDrive/Image_org/Front_view/"+data+".jpg");

for filen in os.listdir("drive/MyDrive/Image_user/Front_view/"):
    if(data + ".jpg" in filen):
      imga = plt.imread("drive/MyDrive/Image_user/Front_view/"+data+".jpg");
```

Here above mentioned orb and structural similarity func are used to generate score 

```js
data_f = []
data_of=[]

orb_similarity = orb_sim(img,imga)
data_of.append(orb_similarity)

data_sf = []
img_b = np.squeeze(img)
img_c = np.squeeze(imga)
ssim = structural_sim(img_b, img_c)
data_sf.append(ssim)
```

Finally the avg is taken for both similarity scores and saved to particular location

```js
maxi = data_of[0];    
print(maxi ,":", data)

maxt = data_sf[0];    
print(maxt ,":", data)

avga = (maxt+maxi)/2;
with open('drive/MyDrive/Image_output/Front_score/out.txt', 'w') as f:
    print(avga,":", data, file=f)
```

### Audio Comparison

Similarly, audio file matching the name(brand_model) will be searched in the database for both accelerating and deaccelerating audio, comparison will be done using audio fingerprinting (implemented using chromprint) and score for each comparison will be generated. Avarage for both the scores will be taken

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/f485b2b3-1f50-4fa1-8512-ef9e504b69ed)



Aduio finger printing is done here

```js
import acoustid
import chromaprint
duration, fp_encoded = acoustid.fingerprint_file(file1)
fingerprint1, version = chromaprint.decode_fingerprint(fp_encoded)
duration, fp_encoded2 = acoustid.fingerprint_file(file2)
fingerprint2, version = chromaprint.decode_fingerprint(fp_encoded2)
print(fingerprint1)
print(fingerprint2)
```

For plotting above fingerprints

```js
import numpy as np
import matplotlib.pyplot as plt
fig1 = plt.figure()
bitmap1 = np.transpose(np.array([[b == '1' for b in list('{:32b}'.format(i & 0xffffffff))] for i in fingerprint1])) #Int to unsigned 32-bit array
plt.imshow(bitmap1)

fig2 = plt.figure()
bitmap2 = np.transpose(np.array([[b == '1' for b in list('{:32b}'.format(i & 0xffffffff))] for i in fingerprint2]))
plt.imshow(bitmap2)
```

Fuzz ratio is used for generating a score based similarity

```js
similarity2 = fuzz.ratio(fingerprint3, fingerprint4)
print(similarity2)
```

## Valuation Equation

Finally the second hand price will be evaluated using a valuation equation:
```js
js Y2=(old_price-((x1*yo)+(x2*ow)+(x3*lo)+(x4*km)+(x5*(1-image_sc))+(x6*(1-audio_sc))))
```
where each coffecient is assigned weights according to how much influnce does it have on the vehicles wear n tear. (image_sc and audio_sc tells how similar images and audio are in comparison to user data, therefore 1-is done to get the dissimilarity)

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/b9a2f35e-6bbe-438e-870d-9dfad06ba231)


## Training the Model

Now we want our model to predict this score even faster, i.e to avoid equation computaion time we'll train our data on ml(XGBoost) model. So once our databe has a minimum of 1000 enteries(model has generated output for atleast 1000 user's vehicle) then automatically our model will start using our XGBoost model but since 
our project was on the testing phase we generated synthetic samples using smote func.

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/0a7b2922-bed4-4557-9775-323a38dd87e5)

This code below is how we generated synthetic samples using SMOTE func

```js
import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors

def SMOTE(T, N, k):
    n_minority_samples, n_features = T.shape

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = int(N/100)
    n_synthetic_samples = int(N * n_minority_samples)
    n_features = int(n_features)
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i].reshape(1, -1), return_distance=False)
        for n in range(N):
            nn_index = choice(nn[0]) 
            while nn_index == i:
                nn_index = choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]

    return S

fd = fd.to_numpy()
new_data = SMOTE(fd,10000,10)
```

Once a sufficent amount of synthetic samples were generated our model successfully executed the shift from scoring to predicting the score through ml model and generated a price almost similar to the one generated through non ml model


