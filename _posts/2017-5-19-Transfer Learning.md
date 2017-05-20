---
layout: post
title: Transfer Learning
comments: true
---

Andrew Ng said during his widely popular NIPS 2016 tutorial that transfer learning, after supervised learning, will be the next driver of ML commercial success.

![NG](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/andrew_ng_drivers_ml_success-1.png "NG")

### What is Transfer Learning?

Transfer learning is the transferring of knowledge gained from one model(trained on a significantly larger dataset) to another dataset with similar characteristics. For example, <a href="http://www.image-net.org/">Imagenet</a> contains images for 1000 categories. It is a competition held every year and VGG-16, Resnet50, InceptionV3, etc models were invented in this competition. 

If we use these models on say, <a href="https://www.kaggle.com/c/dogs-vs-cats">Dogs vs Cats</a> we would be using transfer learning. Since, Imagenet already has images of dogs and cats we would just be using the features learned by the models trained on Imagenet for our purpose.

Transfer learning was used in detecting <a href="http://news.stanford.edu/2017/01/25/artificial-intelligence-used-identify-skin-cancer/">skin cancer</a>. This paper was in the Nature magazine.

### Why Transfer Learning?

In practice, very few people train their own convolutional net from scratch because they don't have sufficient data. **It is always recommended to use transfer learning in practice.**

