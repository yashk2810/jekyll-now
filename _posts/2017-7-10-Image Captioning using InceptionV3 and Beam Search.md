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

For example, Let the image encoding be *IE* and the caption for *IE* is **`<start>` A dog is running in the grass . `<end>`**

<br />
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

## Model
* For image model, I am using a dense layer with input dimension = 2048 which is repeated. 

* For the caption model, I am using a 300 dimensional embedding followed by a LSTM layer with return sequences set to True which is followed by a TimeDistributed Dense layer. 

* For the decoder, I am merging the image model and the caption model which is then put through a Birectional LSTM and Dense layer with 8256 hidden neurons and softmax activation.

```python
embedding_size = 300

image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])
    
caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])
    
final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])
    
final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)
```

**After training it for approximately 35 epochs, the loss value drops down to 2.8876**.

## Predictions

I have used 2 methods for predicting the captions.
* **Max Search** is where the maximum value index(argmax) in the 8256 long predicted vector is extracted and appended to the result. This is done until we hit `<end>` or the maximum length of the caption.

```python
def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])
```

* **Beam Search** is where we take **k** top predictions, feed them again in the model and then sort them using the probabilities returned by the model. 

```python
def beam_search_predictions(image, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption
```

Beam search with k=3 *usually* perform the best.

Finally, here are some results that I got. The rest of the results are in the jupyter notebook and you can generate your own by writing some code at the end.

![two](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/first%202%20images.jpeg "two")
![three](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/3%20images.jpeg "three")
![last](https://raw.githubusercontent.com/yashk2810/yashk2810.github.io/master/images/last.jpeg "last")

<br />
Until next time,

Ciao

<a href="https://twitter.com/share" class="twitter-share-button" data-size="large" data-text="Check out this AWESOME article" data-lang="en" data-show-count="false">Tweet</a><script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
