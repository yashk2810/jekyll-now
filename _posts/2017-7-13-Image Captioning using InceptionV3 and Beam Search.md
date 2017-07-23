---
layout: post
title: Image Captioning using InceptionV3 and Beam Search
comments: true
---

Image Captioning is the technique in which automatic descriptions are generated for an image. In this blog post, I will tell you about the choices that I made regarding which pretrained network to use and how batch size as an hyperparameter can affect your training process.  

**If you directly want to jump to the code, you can find it <a href="https://github.com/yashk2810/Image-Captioning">here</a>**.
The entire code is in the jupyter notebook, so that should hopefully make it easier to understand. I will follow a code first approach and will explain some parts of the code in this post.

## Dataset 

I have used <a href="https://illinois.edu/fb/sec/1713398">Flickr8k dataset</a>(size 1 GB). MS-COCO and Flickr30K are other datasets that you can use. Flickr8K has 6000 training images, 1000 validation images and 1000 testing images. Each image has 5 captions describing it.

I have written the code for MS-COCO but haven't run the model because I am a student right now and it is expensive for me to run a model for weeks. So if you have the resources, you can run the model. The code for the MS-COCO dataset is not clean but you can find the relevant parts and run it. Download the dataset and the captions <a href="http://mscoco.org/dataset/#download">here</a>. The notebook for MS-COCO lives <a href="https://www.dropbox.com/s/zpndo8pdknoqk5k/MS-COCO%20InceptionV3.ipynb">here</a>.

## Image feature extraction

![karpathy](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/karpathy.jpg "karpathy")

This image is taken from the slides of CS231n Winter 2016 Lesson 10 Recurrent Neural Networks, Image Captioning and LSTM taught by Andrej Karpathy.

In Image Captioning, a CNN is used to extract the features from an image which is then along with the captions is fed into an RNN. To extract the features, we use a model trained on Imagenet. I tried out VGG-16, Resnet-50 and InceptionV3. Vgg16 has almost 134 million parameters and its top-5 error on Imagenet is 7.3%. InceptionV3 has 21 million parameters and its top-5 error on Imagenet is 3.46%. Human top-5 error on Imagenet is 5.1%. 

* I used VGG-16 as my first model for extracting the features. I took *an hour* to extract features from 6000 training images. This is very slow. Imagine how much time it will take to extract features in the MS-COCO dataset which has 80,000 training images.

* Resnet-50 was the second model I tried for extracting features. But I didn't train the model for long time because InceptionV3 has a better accuracy than Resnet-50 and almost the same number of parameters.

* Finally, it was the time of InceptionV3. Since it has very less parameters as compared to VGG-16, it took *20 mins* for InceptionV3 to extract features from 6000 images. I also ran this on MS-COCO dataset which contains 80,000 training examples and it took *2 hours and 45 minutes* to extract the features.

## Training and Hyperparameters

For creating the model, the captions has to be put in an embedding. I wanted to try Word2Vec to get the pre-trained embedding weights of my vocabulary, but it didn't pan out. So, I took some ideas from it by setting the embedding size to 300. The image below is the model that I used.

![final_model](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/final_model.jpg "final_model")

* The optimizer used was RMSprop and the batch size was set to 128.

* I trained the model using the VGG-16 extracted features for about 50 epochs and got a loss value of **2.77**.

* After training the model using the InceptionV3 extracted features for about 35 epochs and got a loss value of **2.8876**. 
  * I trained this model for another 15 epochs and the loss value was not going below 2.88.
  
  * I tried learning rate annealing, changing the optimizer, changing the model architecture, the embedding size, number of LSTM units and almost every other hyperparameter. I was stuck.
  
  * One evening while listening to Hans Zimmer's Interstellar track, No time for Caution, it struck me that there was one hyperparameter that I hadn't tried. It was, wait for it... **BATCH SIZE**.
  
  * So, I changed the batch size from 128 to 256 and *voila*, the loss dropped to **2.76** beating VGG-16 at 36th epoch(I got 2.8876 at 35th epoch and I trained the model from there only).
  
  * Whenever the loss started to flatten out, I would double my batch size and the loss started to decrease again. I reached a batch size of 2048 and tried going to 4096 but got a *memory error*.
  
  * The final loss value I got was **1.5987** after training it for **50 epochs**.
  
  * The reason changing the batch size worked was because if the batch size is small, the **gradients are an approximation of the real gradients**. So, it will take longer to find a good solution. If I would have trained the model for another 100 epochs at 128 as my batch size, hopefully the loss would have decreased. 
  
  * Moreover, increasing my batch size decreased by training time. First at batch size of 128 it took approximately 1000 seconds for an epoch. With a batch size of 2048, it took me 343 seconds per epoch.
  
  * So if you are stuck in a similar situation, try increasing the batch size.

## Predictions

I have used 2 methods for predicting the captions.
* **Argmax Search** is where the maximum value index(argmax) in the 8256 long predicted vector is extracted and appended to the result. This is done until we hit `<end>` or the maximum length of the caption.

<script src="https://gist.github.com/yashk2810/5d7cdeca9d5bbf9d4e2b80d7a0d3d256.js"></script>

* **Beam Search** is where we take top **k** predictions, feed them again in the model and then sort them using the probabilities returned by the model. So, the list will always contain the top **k** predictions. In the end, we take the one with the highest probability and go through it till we encounter `<end>` or reach the maximum caption length. 

<script src="https://gist.github.com/yashk2810/f14671f6ad2453d6b7fe029095bfeb84.js"></script>

Finally, here are some results that I got. The rest of the results are in the jupyter notebook and you can generate your own by writing some code at the end.

!["first2"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/first2.jpg "first2")
!["second2"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/second2.jpg "second2")
!["third"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/third.jpg "third")
!["last1"](https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/last1.jpg "last1")


## Next Steps

Attention has been proven to be very effective in a plethora of tasks including image captioning. Implementing attention in my model will definitely lead to an improvement. If anyone of you are interested in this you can send a pull request. 

<br />
*If you’re interested in collaborating, discussing or working with me on an exciting idea, contact me at yash DOT katariya10 AT gmail.com*

<br />
Until next time,

Ciao

<a href="https://twitter.com/share" class="twitter-share-button" data-size="large" data-text="Check out this AWESOME article" data-lang="en" data-show-count="false">Tweet</a><script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
