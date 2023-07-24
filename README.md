# TVS_Epic_IT_Challenge

This project was made in collaboration with @HarshGupta02 (https://github.com/HarshGupta02) for tvs_it_challenge hackathon

In this we were suppsed to automate the entire evaluation process of used automobiles such that no human intervention would be required in between.
For that we devided our project into client side and server side.

## Installation

Since ml part is done on colab most of libraries will be pre-installed, here are few which still needs to be included  

1. pip install pyacoustid
2. pip install fuzzywuzzy[speedup]
3. pip install typing

## Client Side

This is basically the frontend of our website where client/user will enter all the information regarding his vehicle like Brand,Year,Model_name,Ownershipn and Location and then to make the evaluation even better we asked users to enter their vehicle's accelerating and deaccelerating sounds and its 4 sided view for audio and image comparison as well

![WhatsApp Image 2023-07-24 at 17 38 35](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/f4661e72-fc08-4132-b09b-62573a869aac)

## Server Side

This is where all the comparisons are done in the backendand and score generated will be shown in the fronted. It heppens in few stpes:

-> For image comparison, images provided by the user will be located in the database by their brand_model name and 4-sided view comparisomn will be done with the original(new) pics of that brand_model using orb and structural similarity. Score for each view comparison will be generated and avg of them is taken.

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/81a30d51-3c8c-4b52-87e6-e0f18e8c3b5b)

-> Similarly, audio file matching the name(brand_model) will be searched in the database for both accelerating and deaccelerating audio, comparison will be done using audio fingerprinting (implemented using chromprint) and score for each comparison will be generated. Avarage for both the scores will be taken

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/f485b2b3-1f50-4fa1-8512-ef9e504b69ed)

->Then second hand price will be evaluated using a valuation equation _Y=(old_price-((x1*yo)+(x2*ow)+(x3*lo)+(x4*km)+(x5*(1-image_sc))+(x6*(1-audio_sc))))_  where each coffecient is assigned weights according to how much influnce does it have on the vehicles wear n tear. (image_sc and audio_sc tells how similar images and audio are in comparison to user data, therefore 1-is done to get the dissimilarity)

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/b9a2f35e-6bbe-438e-870d-9dfad06ba231)


Now we want our model to predict this score even faster, i.e to avoid equation computaion time we'll train our data on ml(XGBoost) model. So once our databe has a minimum of 1000 enteries(model has generated output for atleast 1000 user's vehicle) then automatically our model will start using our XGBoost model but since our project was on the testing phase we generated synthetic samples using smote func.

![image](https://github.com/Jarvis-Keshav/Car_Valuations/assets/79581388/0a7b2922-bed4-4557-9775-323a38dd87e5)

Once a sufficent amount of synthetic samples were generated our model successfully executed the shift from scoring to predicting the score through ml model and generated a price almost similar to the one generated through non ml model
