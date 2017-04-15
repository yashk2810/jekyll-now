---
layout: post
title: Applying Convolutional Neural Network on the MNIST dataset
---

Convolutional Neural Networks have changed the way we classify images. It is being used in almost all the computer vision tasks. From 2012, CNN's have ruled the Imagenet competition, dropping the classification error rate each year. MNIST is the most studied dataset (<a href='https://www.kaggle.com/benhamner/d/benhamner/nips-papers/popular-datasets-over-time' target="_blank">link</a>). 

The state of the art result for MNIST dataset has an accuracy of 99.79%. In this article, we will achieve an accuracy of 99.55%.

## What is the MNIST dataset?

![MNIST](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/mnist.png "MNIST")

MNIST dataset contains images of handwritten digits. It has 60,000 grayscale images under the training set and 10,000 grayscale images under the test set. We will use the Keras library with Tensorflow backend to classify the images.

## What is a Convolutional Neural Network?

A convolution in CNN is nothing but a element wise multiplication i.e. dot product of the image matrix and the filter.

![Convolution](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/convolution.gif "Convolution")

In the above example, the image is a 5 x 5 matrix and the filter going over it is a 3 x 3 matrix. A convolution operation takes place between the image and the filter and the convolved feature is generated. Each filter in a CNN, learns different characteristic of an image. 

## Implementation

First, we import all the necessary libraries required.

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
{% endhighlight %}


The MNIST dataset is provided by Keras.
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
The shape of X_train is (60000, 28, 28). Each image has 28 x 28 resolution. 
The shape of X_test is (10000, 28, 28).

The input shape that a CNN accepts should be in a specific format. If you are using Tensorflow, the format should be (batch, height, width, channels). If you are using Theano, the format should be (batch, channels, height, width).

So, let's reshape our input.

```python
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255
```
Now the shape of X_train is (60000, 28, 28, 1). As all the images are in grayscale, the number of channels is 1. If it was a color image, then the number of channels would be 3 (R, G, B).

Here we’ve rescaled the image data so that each pixel lies in the interval [0, 1] instead of [0, 255]. It is always a good idea to normalize the input so that each dimension has approximately the same scale.

Now, we need to one-hot encode the labels i.e. Y_train and Y_test. In one-hot encoding an integer is converted to an array which contains only one '1' and the rest elements are '0'. Why do we need to one-hot encode the labels? We will come back to it.

```python
number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
```
Y_train[0] = [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.] since the label representated by it is 5.  
<br />

Let's create the model that will classify the images (the most interesting part!!).

```python
# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))
```




