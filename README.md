# greenlife  
Leaf Health Image Classifier  
Project Date: April 2017  

<img src="https://github.com/giancarlo-garbagnati/greenlife/raw/master/images/Cherry-healthy-00032.JPG" alt="Cherry (healthy) Leaf (from PlantVillage)" width="130" height="200"/>  

## Description  

PlantVillage recently [hosted](https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge) a challenge to build an image classification model for plant images that could distinguish between crop species and can accurately diagnose if the plant has a disease (and which one) (for background on the dataset, refer to https://arxiv.org/abs/1604.03169). While I was too late to participate in the challenge, I still used the dataset to build this model. It was made using transfer learning of the convolutional neural network [InceptionV3](https://arxiv.org/abs/1512.00567) architecture. After training through 50+25 epochs (50 epochs training fully connected layer, and 25 epochs 'fine-tuning' the top 2 inception modules), the image classification model had over 90% accuracy, which is a credit to the power of deep-learning and to the folks at Google.  

## Data Acquisition  

PlantVillage had hosted the images and [csv's](https://github.com/giancarlo-garbagnati/greenlife/tree/master/image_csv) with all the images and classes. From this, it was pretty straight-forward to write a [script](https://github.com/giancarlo-garbagnati/greenlife/blob/master/python_scripts/image_scraper.py) to download all the images listed in the csvs.

## Dataset  

Once all the images were downloaded, the crops that didn't have both healthy and diseased image sets were removed from the final model. In the end, there was over 70000 images over 46 total 'categories' used in the model:  
* 14 healthy crop categories (apple, banana, bell pepper, cabbage, cherry, corn, cucumber, grape, peach, potato, soybean, squash, strawberry, tomato)  
* 32 diseased crop categories  



## Modeling/Results  
