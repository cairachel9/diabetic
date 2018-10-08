# Introduction 

<img align = "right" src="image/NIH.png" width="60" height="40" />

During the summer of 2018 and the following school year, I had the opportunity to work in **National Institute of Health (NIH)** Lister Hill Center to study convolutional neural network for early detection of diabetic retinopathy eye disease.  The project was conducted under the mentorship of Dr. Jongwoo Kim.

## Diabetic retinopathy

[Diabetic retinopathy](http://en.wikipedia.org/wiki/Diabetic_retinopathy) is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.

<img align = "left" src="image/1000_left.jpeg" width="258" height="172" />

Around 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.

Presently, detecting DR is a manual time-consuming process that requires a trained clinician to examine and evaluate digital fundus photographs of the retina. By the time human readers submit their reviews, sometimes several days later, the delayed results lead to lost miscommunication, follow up, and delayed treatment.

## Computer Vision through Convolutional Neural Network

Convolutional Neural Networks (CNNs), a part of deep learning, have a great record for applications in image analysis and interpretation, including medical imaging. However, it wasn’t until several breakthroughs in neural networks such as the implementation of dropout, rectified linear units and the accompanying increase in computing power through graphical processor units (GPUs) that they became viable for more complex image recognition problems. Presently, large CNNs are used to successfully tackle highly complex computer vision tasks with many object classes to an impressive standard. CNNs are used in many current state-of-the-art image classification tasks such as the annual ImageNet challenges.

## CNN for Diabetic Retinopathy detection

[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) is feed-forward neural network.  It mainly consists of an input layer, many hidden layers (such as convolutional relu, pooling, flatten, fully connected and softmax layers) and a final multi-label classificationlayer. CNN methodology involves two stages of processing, a time consuming training stage where millions of images went through CNN architecture on many iterations to finalize the parameters of each layer (finalize the model parameters) and a second real-time prediction stage where each image in test dataset is feeded into the trained model to score and validate the model.

<img src="image/CNN_DR.png" width="750" height="200" />

The output of the above framework will emit a multi-class prediction with confidence score on each category
* 65% No DR (Normal)
* 15% Category-2 DR
* 10% Category-4 DR

However, there are two issues with CNN methods on DR detection. One is achieving a desirable offset in sensitivity (patients correctly identified as having DR) and specificity (patients correctly identified as not having DR). This is significantly harder for a five class problem of normal, mild DR, moderate DR, severe DR, and proliferative DR classes. Second is the overfitting problem. Skewed datasets cause the network to over-fit to the class most prominent in the dataset. Large datasets are often massively skewed.

# Our Work

We explored the use of deep convolutional neural network methodology for the automatic classification of diabetic retinopathy using color fundus image.  We evaluated several CNN architectures and studied various tuning techniques in model training.

## Methodology

In this project, we evaluated several CNN architecture (Inception Network, VGG16) on their performance in DR image recognition.  We applied various standard techniques to cleanse and augment the data, we also optimized the CNN network to accomodate the skewed data sets.

Our experiment was conducted on the Linux platform with NVidia Tesla K80 GPU.  The environment was hosted by Google Colab and Kaggle.

### Datasets
#### Kaggle DR competition dataset

Our main dataset is based on the [Kaggle Diatebic Retinopathy Detection competition] (https://www.kaggle.com/c/diabetic-retinopathy-detection) carried out in 2016.  The main dataset contains 35000 eye images with 5 stages of DR disease.

#### Messidor dataset

We also augmented the dataset with [Messidor dataset](http://www.adcis.net/en/Download-Third-Party/Messidor.html) which contains 1200 images with 4 stage DR progression.  Although the Messidor dataset is smaller, it has less labeling errors.

### Stages of diabetic retinopathy (DR) with increasing severity
The following figures show the 5 class DR classification in our study, range from DR_0 (No DR) to DR_5 Proliferative DR (Proliferative DR).

<img src="image/5_DR.png" width="750" height="300" />

#### Unbalanced training data set
<img align="left" src="image/level_unbalanced.png" width="200" height="200" />

Skewed datasets cause the network to over-fit to the class most prominent in the dataset. Large datasets are often massively skewed. 

In the Kaggle dataset with 35000 images, we used less than three percent of images came from the 4th and 5th class, meaning changes had to be made in our network to ensure it could still learn the features of these images.  To overcome the difference in data points distribution, we used the the sampling with [replacement statistics technique](https://web.ma.utexas.edu/users/parker/sampling/repl.htm) to boost up the data samples in category 2, 4 and 5:

For the eye distribution between left and right eye, we have a balanced distribution. 

<img src="image/level_balanced.png" width="600" height="250" />

### CNN Architectures

There are various CNN architecutres proposed in the academia: VGG16, Inception, ResNet, GoogLeNet.  [This pyImageSearch article](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/) has a good introduction to many poppular CNN architectures.

#### InceptionV3
<img src="image/InceptionV3.png" width="600" height="220" />

The Inception microarchiveture was first introduced by Szegedy et al. in their 2014 pagper [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842).  The goal of the inception module is to act as "multi-level feature extractor" by computing 1x1, 3x3 and 5x5 convolutions within the same module of the network.  The original incarnation of this architecture was called GoogLeNet, but subsequent manifestations have simply been called Inception vN where N is the version number put out by Google.

The weights for Inception V3 are smaller than both VGG and ResNet coming in at size of 96MB.

#### VGG16 and VGG19
<img align="left" src="image/imagenet_vgg16.png"/>

The VGG network architecture was introduced by Simonyan and Zissermain in their 2014 paper: [Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/abs/1409.1556). VGG network is characterized by its simplicity, using only 3x3 convolutional layers stacked on top of each other in increasing depth.  Reducing volumn size is handled by max pooling.  Two fully-connected layers, each with 4096 nodes are then followed by a softmax classifier (above).  The "16" and "19" stands for the number of weight layers in the network.

The drawbacks for VGGNetwork are one it is slow to train and the network architecture weights themselves are quite large large.

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
