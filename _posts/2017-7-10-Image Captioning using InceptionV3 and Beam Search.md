---
layout: post
title: Image Captioning using InceptionV3 and Beam Search
comments: true
---

In lesson 8 of Practical Deep Learning For Coders, Part 2 by Jeremy Howard and Rachel Thomas, Jeremy encouraged everyone to do a project. I didn't attend the course in-person and watched the videos when they were released(still in preview) on the forums. 
So, I asked Jeremy whether Image Captioning was implemented in this part of the course and he suggested that it would be a good class project. Hence, I decided that image captioning would be my project.

If you directly want to jump to the code, you can find it <a href="https://github.com/yashk2810/Image-Captioning">here</a>.
The entire code is in the jupyter notebook, so that should hopefully make it easier to understand. I will follow a code first approach and will explain some parts of the code in this post.

## Dataset 

I have used <a href="https://illinois.edu/fb/sec/1713398">Flickr8k dataset</a>(size 1 GB). MS-COCO, Flickr30K are other datasets that you can use. Flickr8K has 6000 training images, 1000 validation images and 1000 testing images. Each image has 5 captions describing it.

I have written the code for MS-COCO but haven't run the model because frankly I don't have the money to run the model for weeks on the AWS p2 instance and neither do I own a DL box. So if you have the resources, you can run the model. The code for the MS-COCO dataset is not clean but you can find the relevant parts and run it. Download the dataset and the captions <a href="http://mscoco.org/dataset/#download">here</a>. The notebook for MS-COCO lives <a href="https://www.dropbox.com/s/zpndo8pdknoqk5k/MS-COCO%20InceptionV3.ipynb">here</a>.

## Image feature extraction

!["karpathy"](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/karpathy.jpg "karpathy")

In Image Captioning, a CNN is used to extract the features from an image which is then along with the captions is fed into an RNN. To extract the features, we use a model trained on Imagenet. I tried out VGG16, Resnet-50 and InceptionV3. 



