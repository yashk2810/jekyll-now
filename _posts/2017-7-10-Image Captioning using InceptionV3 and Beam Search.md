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

![karpathy](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/karpathy.jpg "karpathy")

This image is taken from the slides of CS231n Winter 2016 Lesson 10 Recurrent Neural Networks, Image Captioning and LSTM taught by Andrej Karpathy.

In Image Captioning, a CNN is used to extract the features from an image which is then along with the captions is fed into an RNN. To extract the features, we use a model trained on Imagenet. I tried out VGG16, Resnet-50 and InceptionV3. Vgg16 has almost 134 million parameters and its top-5 error on Imagenet is 7.3%. InceptionV3 has 21 million parameters and its top-5 error on Imagenet is 3.46%. Human top-5 error on Imagenet is 5.1%. 

Since InceptionV3 has less parameters and a greater accuracy, I decided to use InceptionV3 to extract features from an image.

```python
# This is how the preprocessing was done
# by the Inception authors
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
 
# The image size used was 299 X 299.
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)
```

If an image is fed into "model_new", we will get a numpy array of shape **(1, 2048)**. InceptionV3 doesn't have any fully connected layers, instead it has Average pooling layer which is the reason of less parameters. VGG16's first fully connected layer contributes 102 million parameters out of the 134 million parameters.

Now, we can use *model_new* to extract the features from all our training images.

## Data Generator



