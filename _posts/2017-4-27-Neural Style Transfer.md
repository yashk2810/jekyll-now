---
layout: post
title: Neural Style Transfer
comments: true
---

![Baby Flower](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/baby_flower.png "Baby Flower")

Neural Style Transfer is the process in which content of an image and style of another image are combined together to create a new image. Prisma uses style transfer to transform your photos into works of art using style of famous artists.

The GIF above was created using style transfer. In this blog post, we will learn how to implement it and reproduce these amazing results. We will use Keras with Tensorflow backend to achieve this. 

## Import the necessary libraries
```python
import matplotlib.pyplot as plt
%matplotlib inline
import importlib
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from keras import metrics
from keras.model import Model
from keras.applications import VGG16
from PIL import Image
import keras.backend as K
```

## Recreate Content

Before we do style transfer, we need to recreate the content of the image from a random image. The content image can be found <a href="https://github.com/yashk2810/yashk2810.github.io/blob/master/images/hugo.jpg">here</a>

```python
img = Image.open('hugo.jpg')
img.size
```

We will use the VGG16 model(pre-trained on Imagenet) to get the activations necessary to calculate the target activations.
In the VGG16 model, the authors subtracted the mean of each channel(R, G, B) from the image and the channel format used was BGR instead of RGB. The mean of each channel has been provided by the authors of the VGG16 model.

Hence, we need to do the necessary preprocessing of the image before we use the model.

```python
mean_sub = np.array([123.68, 116.779, 103.939], dtype=np.float32)
pre_processing = lambda x: (x - mean_sub)[:,:,:,::-1]
```

To plot the image again, we need to deprocess it.

```python
de_preprocess = lambda x, shape: np.clip(x.reshape(shape)[:,:,:,::-1] + mean_sub, 0, 255)
```

Let's convert the image into (batch_size, height, width, channels) so we can feed it to the CNN.

```python
img_arr = pre_processing(np.expand_dims(np.array(img), 0))
shape_content = img_arr.shape
```









