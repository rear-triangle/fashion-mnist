# fashion-mnist

SETUP: -create a folder called 'large_files' (or modify the name in the script to your working folder)
- cd into the folder you created
- download the fashion-mnist dataset from Kaggle with the following link: 'https://www.kaggle.com/zalando-research/fashionmnist' and move it into the folder specified above
- run the script and allow time for training

Since many image classification models have used the hand-written digit dataset as their 'Hello World' dataset, many models may have become too well adjusted to the dataset. This repository will use fashion-mnist as the dataset to be proactive about any unintended impact to accuracy resulting from bias, which instead has the model try to classify between greyscale images of articles of clothing.

The choice to create three convolution layers was somewhat arbitrary, as diminishing returns in accuracy are likely to be expected past 2 or 3 convolutional layers. 

Since we are dealing with multiple instead of binary classification, we use categorical crossentropy instead of binary crossentropy for the loss function and softmax instead of sigmoid for the activation function in the final layer.