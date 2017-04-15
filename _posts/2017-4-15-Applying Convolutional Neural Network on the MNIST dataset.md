---
layout: post
title: Applying Convolutional Neural Network on the MNIST dataset
---

Convolutional Neural Networks have changed the way we classify images. It is being used in almost all the computer vision tasks. From 2012, CNN's have ruled the Imagenet competition, dropping the classification error rate each year. MNIST is the most studied dataset (<a href='https://www.kaggle.com/benhamner/d/benhamner/nips-papers/popular-datasets-over-time' target="_blank">link</a>). 

The state of the art result for MNIST dataset has an accuracy of 99.79%. In this article, we will achieve an accuracy of 99.55%.

## What is the MNIST dataset?

![MNIST](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/mnist.png "MNIST")

MNIST dataset contains images of handwritten digits. It has 60,000 images under the training set and 10,000 images under the test set. We will use the Keras library with Tensorflow backend to classify the images.

## What is a Convolutional Neural Network?

A convolution in CNN is nothing but a element wise multiplication i.e. dot product of the image matrix and the filter.

![Convolution](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/convolution.gif "Convolution")

