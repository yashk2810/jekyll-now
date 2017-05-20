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

The below images show the things that convolutional networks learn when trained on Imagenet and why it is effective to use transfer learning.

![deconvnet1](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/deconvnet1.png "deconvnet1")
![deconvnet2](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/deconvnet2.png "deconvnet2")

As you can see in layer 1 and layer 2 visualization, the conv net learned to recognize edges, circles, etc. But the layer 4 and layer 5 learn to recognize the entire objects like dogs face, umbrella, etc. 

So, if we use a model trained on imagenet for Dogs vs Cats prediction we can just change the last classifier(fully-connected) layer since the last convolutional layer already knows what a dog or a cat looks like.

What if the dataset is not similar to the pretrained model data? In such cases, we can train the convnet by finetuning the weights of the pretrained model by continuing the backpropogation. You can finetune the entire convnet or keep some early layers of the network fixed(non-trainable) and finetune the higher layers. This is because the early layers contain general information about the image but the later layers become more specific to the classes in the original dataset.


