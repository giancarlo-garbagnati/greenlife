# greenlife  
Leaf Health Image Classifier  
Project Date: April 2017  

<img src="https://raw.githubusercontent.com/giancarlo-garbagnati/greenlife/master/Cherry-healthy-00032.JPG" alt="Apple Leaf (from PlantVillage)" width="130" height="200"/>  

## Description  

PlantVillage recently [hosted](https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge) a challenge to build an image classification model for plant images that could distinguish between crop species and can accurately diagnose if the plant has a disease (and which one) (for background on the dataset, refer to https://arxiv.org/abs/1604.03169). While I was too late to participate in the challenge, I still used the dataset to build this model. It was made using transfer learning of the convolutional neural network [InceptionV3](https://arxiv.org/abs/1512.00567) architecture. After training through 50+25 epochs (50 epochs training fully connected layer, and 25 epochs 'fine-tuning' the top 2 inception modules), the image classification model had over 90% accuracy, which is a credit to the power of deep-learning and to the folks at Google.

## Data Acquisition  

## Dataset  

## Modeling/Results  
