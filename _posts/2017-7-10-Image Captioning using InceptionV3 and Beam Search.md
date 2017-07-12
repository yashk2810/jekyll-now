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

After extracting the features, we need to calculate the vocabulary size. So we get all the unique words from the training captions and create vocabulary. The vocabulary size is **8256**. The code for it is trivial. You can refer the notebook for that.

While passing the data to the model, we need to pass the image encoding and the first word as the input and the output will be the second word. Next input will be the image encoding, first word and the second word and the output will the third word. This will go on for the entire caption and then the second image encoding will come and its captions and so on.

We will add 2 more words *`<start>`* and *`<end>`* to identify the starting and ending of a sentence. This will be useful when we have to decrypt the predictions.

**NOTE:- The captions that are fed to the model are not words but indices of those words stored in our vocabulary. In the code, I have created 2 dictionaries; word2idx(word to index) and idx2word(index to word).**

So, in the example below, the input won't be **"dog"** instead it will be **word2idx["dog"].**

For example, Let the image encoding be *IE* and the caption for *IE* is "`<start>` A dog is running in the grass . `<end>`"
![table](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/table.jpg "table")

So this is how we will give the input to the model and it has to predict 1 word out of 8256 words.
Because of such input we will need to design our own generator and give the input to the model batch-wise.

```python
# Our custom data generator
def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        # shuffling the dataframe every epoch
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0
```



