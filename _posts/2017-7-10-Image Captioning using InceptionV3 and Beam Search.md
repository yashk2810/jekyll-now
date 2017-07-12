---
layout: post
title: Image Captioning using InceptionV3 and Beam Search
comments: true
---

In lesson 8 of Practical Deep Learning For Coders, Part 2 by Jeremy Howard and Rachel Thomas, Jeremy encouraged everyone to do a project. I didn't attend the course in-person and watched the videos when they were released(still in preview) on the forums. 
So, I asked Jeremy whether Image Captioning was implemented in this part of the course and he suggested that it would be a good class project. Hence, I decided that Image Captioning would be my project. 

If you directly want to jump to the code, you can find it <a href="https://github.com/yashk2810/Image-Captioning">here</a>.
The entire code is in the jupyter notebook, so that should hopefully make it easier to understand. I will follow a code first approach and will explain some parts of the code in this post.

## Dataset 

I have used <a href="https://illinois.edu/fb/sec/1713398">Flickr8k dataset</a>(size 1 GB). MS-COCO and Flickr30K are other datasets that you can use. Flickr8K has 6000 training images, 1000 validation images and 1000 testing images. Each image has 5 captions describing it.

I have written the code for MS-COCO but haven't run the model because I am a student right now and it is expensive for me to run a model for weeks. So if you have the resources, you can run the model. The code for the MS-COCO dataset is not clean but you can find the relevant parts and run it. Download the dataset and the captions <a href="http://mscoco.org/dataset/#download">here</a>. The notebook for MS-COCO lives <a href="https://www.dropbox.com/s/zpndo8pdknoqk5k/MS-COCO%20InceptionV3.ipynb">here</a>.

## Image feature extraction

![karpathy](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/karpathy.jpg "karpathy")

This image is taken from the slides of CS231n Winter 2016 Lesson 10 Recurrent Neural Networks, Image Captioning and LSTM taught by Andrej Karpathy.

In Image Captioning, a CNN is used to extract the features from an image which is then along with the captions is fed into an RNN. To extract the features, we use a model trained on Imagenet. I tried out VGG16, Resnet-50 and InceptionV3. Vgg16 has almost 134 million parameters and its top-5 error on Imagenet is 7.3%. InceptionV3 has 21 million parameters and its top-5 error on Imagenet is 3.46%. Human top-5 error on Imagenet is 5.1%. 

Since InceptionV3 has less parameters and a greater accuracy, I decided to use InceptionV3 to extract features from an image.

<script src="https://gist.github.com/yashk2810/47cd94f27003e8926dde98d24058b781.js"></script>

If an image is fed into "model_new", we will get a numpy array of shape **(1, 2048)**. InceptionV3 doesn't have any fully connected layers, instead it has Average pooling layer which is the reason of less parameters. VGG16's first fully connected layer contributes 102 million parameters out of the 134 million parameters.

Now, we can use *model_new* to extract the features from all our training images.

## Data Generator

After extracting the features, we need to calculate the vocabulary size. So we get all the unique words from the training captions and create vocabulary. The vocabulary size is **8256**. The code for it is trivial. You can refer the notebook for that.

While passing the data to the model, we need to pass the image encoding and the first word as the input and the output will be the second word. Next input will be the image encoding, first word and the second word and the output will the third word. This will go on for the entire caption and then the second image encoding will come and its captions and so on.

We will add 2 more words *`<start>`* and *`<end>`* to identify the starting and ending of a sentence. This will be useful when we have to decrypt the predictions.

**NOTE:- The captions that are fed to the model are not words but indices of those words stored in our vocabulary. In the code, I have created 2 dictionaries; word2idx(word to index) and idx2word(index to word).**

So, in the example below, the input won't be **"dog"** instead it will be **word2idx["dog"].**

For example, Let the image encoding be *IE* and the caption for *IE* is **`<start>` A dog is running in the grass . `<end>`**

<br />
![table](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/table.jpg "table")

So this is how we will give the input to the model and it has to predict 1 word out of 8256 words.
Because of such input we will need to design our own generator and give the input to the model batch-wise.

<script src="https://gist.github.com/yashk2810/3d22304eed391adcb2273cd2806718dc.js"></script>

## Model
* For image model, I am using a dense layer with input dimension = 2048 which is repeated. 

* For the caption model, I am using a 300 dimensional embedding followed by a LSTM layer with return sequences set to True which is followed by a TimeDistributed Dense layer. 

* For the decoder, I am merging the image model and the caption model which is then put through a Birectional LSTM and Dense layer with 8256 hidden neurons and softmax activation.

<script src="https://gist.github.com/yashk2810/272649b477fb9c26b39bb1d16c4a7e8f.js"></script>

**After training it for approximately 35 epochs, the loss value drops down to 2.8876**.

## Predictions

I have used 2 methods for predicting the captions.
* **Max Search** is where the maximum value index(argmax) in the 8256 long predicted vector is extracted and appended to the result. This is done until we hit `<end>` or the maximum length of the caption.

<script src="https://gist.github.com/yashk2810/5d7cdeca9d5bbf9d4e2b80d7a0d3d256.js"></script>

* **Beam Search** is where we take top **k** predictions, feed them again in the model and then sort them using the probabilities returned by the model. So, the list will always contain the top **k** predictions. In the end, we take the one with the highest probability and go through it till we encounter `<end>` or reach the maximum caption length. 

<script src="https://gist.github.com/yashk2810/f14671f6ad2453d6b7fe029095bfeb84.js"></script>

Beam search with k=3 *usually* perform the best.

Finally, here are some results that I got. The rest of the results are in the jupyter notebook and you can generate your own by writing some code at the end.

![two](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/first%202%20images.jpeg "two")
![three](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/3%20images.jpeg "three")
![last](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/last.jpeg "last")


## Next Steps

Attention has been proven to be very effective in a plethora of tasks including image captioning. Implementing attention in my model will definitely lead to an improvement. If anyone of you are interested in this you can send a pull request. 

<br />
Until next time,

Ciao

<a href="https://twitter.com/share" class="twitter-share-button" data-size="large" data-text="Check out this AWESOME article" data-lang="en" data-show-count="false">Tweet</a><script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
