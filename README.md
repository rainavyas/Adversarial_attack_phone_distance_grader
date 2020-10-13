# Adversarial_attack_phone_distance_grader
Attempt to deceive the grader

Kyriakopoulos, K., Gales, M., Knill, K. (2017) Automatic Characterisation of the Pronunciation of Non-native English Speakers using Phone Distance Features. Proc. 7th ISCA Workshop on Speech and Language Technology in Education, 59-64, DOI: 10.21437/SLaTE.2017-11.





Model Summary:

When attempting to grade a speaker on their English fluency by only considering their pronunciation, phone distances can be used. An ASR system will output a phone classification for each frame of a candidate's speech, using the mfcc vector input. For each speaker and for every phone (47 in English language) a multivariate Gaussian distribution can be estimated (by calculating a mean vector and covariance matrix from the mfcc vectors assigned to the particular phone for each speaker). To reduce the impact of voice quality or accent, the "distance" between these phones' distributions can be computed. A symmetric KL divergence distance is employed, meaning, we can summarise each speaker's phone distances in a 1128 dimensional feature vector (48 x 47 x 0.5). This vector can be passed through a fully connected deep model to predict a fluency grade (for example on a CEFR scale).


# Adversarial Attack

The aim is to add a small universal noise to the audio waveforms, such that the Grader always predicts a higher grade... i.e. simulating an attack an adversary may attempt on an automated assessment system online. 

# Dependencies

### python3.6 or above
### pip install numpy

### pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html -> using CUDA 10.0

These dependencies cover all the requirements to run all the scripts in this repository.

# File Descriptions

## pkl2pdf.py

* Loads the input data from a pickle file - mfcc vectors (indexed by frame, phone, word, utterance, speaker) for Linguaskill-General Data
* Computes a Gaussian distribution (mean and covariance) by speaker for each of the 47 phones
* Writes numpy arrays to npz file
