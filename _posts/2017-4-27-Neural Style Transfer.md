---
layout: post
title: Neural Style Transfer
comments: true
---

![Baby Flower](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/baby_flower.png "Baby Flower")

Neural Style Transfer is the process in which content of an image and style of another image are combined together to create a new image. Prisma uses style transfer to transform your photos into works of art using style of famous artists.

The GIF above was created using style transfer. In this blog post, we will learn how to implement it. We will use Keras with Tensorflow backend to achieve this. 

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




