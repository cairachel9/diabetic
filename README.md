# Introduction 

<img align = "right" src="image/NIH.png" width="60" height="40" />

During the summer of 2018 and the following school year, I had the opportunity to work in **National Institute of Health (NIH)** Lister Hill Center to study convolutional neural network for early detection of diabetic retinopathy eye disease.  The project was conducted under the mentorship of Dr. Jongwoo Kim.

## Diabetic retinopathy

[Diabetic retinopathy](http://en.wikipedia.org/wiki/Diabetic_retinopathy) is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.

<img align = "left" src="image/1000_left.jpeg" width="258" height="172" />

Around 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.

Presently, detecting DR is a manual time-consuming process that requires a trained clinician to examine and evaluate digital fundus photographs of the retina. By the time human readers submit their reviews, sometimes several days later, the delayed results lead to lost miscommunication, follow up, and delayed treatment.

## Computer Vision through Convolutional Neural Network

Convolutional Neural Networks (CNNs), a branch of deep learning, have an impressive record for applications in image analysis and interpretation, including medical imaging. However, it wasn’t until several breakthroughs in neural networks such as the implementation of dropout, rectified linear units and the accompanying increase in computing power through graphical processor units (GPUs) that they became viable for more complex image recognition problems. Presently, large CNNs are used to successfully tackle highly complex computer vision tasks with many object classes to an impressive standard. CNNs are used in many current state-of-the-art image classification tasks such as the annual ImageNet challenges.

## CNN for Diabetic Retinopathy detection

[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) is feed-forward neural network.  It mainly consists of an input layer, many hidden layers (such as convolutional relu, pooling, flatten, fully connected and softmax layers) and a final multi-label classificationlayer. CNN methodology involves two stages of processing, a time consuming training stage where millions of images went through CNN architecture on many iterations to finalize the parameters of each layer (finalize the model parameters) and a second real-time prediction stage where each image in test dataset is feeded into the trained model to score and validate the model.

<img src="image/CNN_DR.png" width="750" height="200" />

The output of the above framework will emit a multi-class prediction with confidence score on each category
* 65% No DR (Normal)
* 15% Category-2 DR
* 10% Category-4 DR

However, there are two issues with CNN methods on DR detection. One is achieving a desirable offset in sensitivity (patients correctly identified as having DR) and specificity (patients correctly identified as not having DR). This is significantly harder for a five class problem of normal, mild DR, moderate DR, severe DR, and proliferative DR classes. Second is the overfitting problem. Skewed datasets cause the network to over-fit to the class most prominent in the dataset. Large datasets are often massively skewed.

# Our Work

Our experiment was conducted on the Linux platform with NVidia Tesla K80 GPU.  The environment was hosted by Google Colab and Kaggle.

## Methodology

### Datasets
#### Kaggle DR competition dataset
#### Messidor dataset

### Stages of diabetic retinopathy (DR) with increasing severity
The following figures show the 5 class DR classification in our study, range from DR_0 (No DR) to DR_5 Proliferative DR (Proliferative DR).

<img src="image/5_DR.png" width="750" height="300" />

#### Unbalanced training data set
<img align="left" src="image/level_unbalanced.png" width="200" height="200" />

Skewed datasets cause the network to over-fit to the class most prominent in the dataset. Large datasets are often massively skewed. 

In the dataset, we used less than three percent of images came from the 4th and 5th class, meaning changes had to be made in our network to ensure it could still learn the features of these images.  We used the the sampling with replacement statistics technique to boost up the data samples in category 2, 4 and 5:

<img src="image/level_balanced.png" width="600" height="250" />

### CNN Architectures
#### InceptionV3
#### VGG16
### Optimizing CNN
#### Preprocessing
#### Data Augmentation
Five different transformation types are used here, including flipping, rotation, rescaling, shearing and translation. See the following table for details:

|Transformation|Description|
|Rotation|0-360|
|Flipping|0 (without flipping) or 1 (with flipping|
|Shearing|Randomly with angle between -15 and 15|
|Rescaling|Randomly with scaling factor between 1/1.6 and 1.6|
|Translation|Randomly with shift between -10 and 10 pixels|

### Training, Gradient Descent
### Training, Pretrained model
### Evaluation, Attention Map

## Prelimary Results

The work is still ongoing.

# Related Work

https://www.kaggle.com/kmader/inceptionv3-for-retinopathy-gpu-hr/notebook

# References

1. Varun Gulshan, Lily Peng, Marc Coram. "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs" JAMA Network, Decemeber 1, 2016. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45732.pdf
2. Harry Pratt, Frans Coenen, Deborah Broadbent, Simon Harding, Yalin Zheng. "Convolutional Neural Networks for Diabetic Retinopathy" 20th Conference on Medical Image Understanding and Analysis (MIUA 2016) July 25, 2016. https://www.sciencedirect.com/science/article/pii/S1877050916311929
3. Carson Lam, Darvin Yi, Margaret Guo, Tony Lindsey.  "Automated Detection of Diabetic Retinopathy using Deep Learning" Proceedings - AMIA Joint Summits on Translational Science.  May 18, 2018. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/
4. Casanova, Ramon, Santiago Saldana, Emily Y. Chew, Ronald P. Danis, Craig M. Greven, and Walter T. Ambrosius. "Application of Random Forests Methods to Diabetic Retinopathy Classification Analyses." PLOS ONE 9, no. 6 (2014): 1-7. Accessed December 26, 2014. www.plosone.org.
5. Sinthanayothin, C., J.F. Boyce, T.H. Williamson, H.L. Cook, E. Mensah, S. Lal, and D. Usher. "Automated Detection of Diabetic Retinopathy on Digital Fundus Images." Diabetic Medicine 19 (2002): 105-12.
6. Usher, D., M. Dumskyjs, M. Himaga, T.H. Williamson, S. Nussey, and J. Boyce. "Automated Detection of Diabetic Retinopathy in Digital Retinal Images: A Tool for Diabetic Retinopathy Screening." Diabetic Medicine 21 (2003): 84-90.
7. Jaafar, Hussain F., Asoke K. Nandi, and Waleed Al-Nuaimy. "Automated Detection And Grading Of Hard Exudates From Retinal Fundus Images." 19th European Signal Processing Conference (EUSIPCO 2011), 2011, 66-70.
8. "National Diabetes Statistics Report, 2014." Centers for Disease Control and Prevention. January 1, 2014. Accessed December 26, 2014.
9. "Diabetes." World Health Organization. November 1, 2014. Accessed December 26, 2014. http://www.who.int/mediacentre/factsheets/fs312/en/.
