#!/usr/bin/env python
# coding: utf-8

# ### Camera images are often unclear due to factors such as lighting, movement, and the age or quality of the camera, all of which can add "noise" to, or blur, an image. This algorithm attempts to extract a clear image from a distorted one.

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Conv2DTranspose,UpSampling2D,Dropout,RandomRotation
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os
from pathlib import Path

cd = Path.cwd()
directory = os.path.join(cd,r'OneDrive\Desktop\Datasets\img_align_celeba\img_align_celeba')
image_paths = []

for image_path in os.listdir(directory):
    image_paths.append(os.path.join(directory,image_path))
    
image_paths = sorted(image_paths)

test_size = 2000
train_paths = image_paths[:-test_size]
validation_paths = image_paths[-test_size:]


# ### A series of helper functions make up the pipeline that standadizes the size of the images, adds a noise or blurring effect, augments the data (such as through flipping and rotating), and creates a Dataset object.

# In[2]:


height = 256
width = 256
batch_size = 128

def extract_and_resize_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image,expand_animations=False)
    image = tf.image.convert_image_dtype(image,tf.float16)
    image = tf.image.resize(image,(height,width))
    return image

def add_noise(image):
    noise_param = tf.random.uniform((1,),7,10)
    noise = tf.random.normal(shape=(height,width,3),mean=1,stddev=1)/noise_param
    noisy_image = tf.add(image,noise)
    noisy_image = tf.clip_by_value(noisy_image,0,1)
    return noisy_image

@tf.function
def blur_image(image):
    t = np.random.uniform(2,3)
    blurred_image = tfa.image.gaussian_filter2d(image,(7,7),t)
    blurred_image = tf.clip_by_value(blurred_image,0,1)
    return blurred_image

rotation_range = .3
random_rotation = RandomRotation((-rotation_range,rotation_range))
def random_flip(image):
    image = tf.image.random_flip_left_right(image)
    if random.randint(0,12) == 1:
        image = tf.image.flip_up_down(image)
    image = random_rotation(image)
    return image

def make_dataset(paths,function_name,train=True,batch_size=batch_size):
    random.shuffle(paths)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(extract_and_resize_image)
    if train:
        dataset = dataset.map(random_flip)
    if function_name == 'noise':
        dataset = dataset.map(lambda image: (add_noise(image),image))
    elif function_name == 'blur':
        dataset = dataset.map(lambda image: (blur_image(image),image))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ### Some examples of how images are alterred, either through adding noise or blur:

# In[3]:


print(' '*15+'Clean Images:'+' '*34+'Altered Images:')
for _ in range(7):
    path = random.choice(train_paths)
    function = random.choice([add_noise,blur_image])
    image = extract_and_resize_image(path)
    altered_image = function(image)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(altered_image)
    plt.axis('off')
    plt.show()


# ### This model is an autoencoder, which uses an encoder model to condense the images into a smaller-dimensional space and a decoder to reconstruct the image without the distortion. The loss function is based on the structural similarity index measure (SSIM), a metric used for comparing the percived quality of two images.

# In[4]:


filters = 32

def conv_bloc(X,filters,filter_size=(3,3),strides=2):
    X = Conv2D(filters,filter_size,strides=strides,activation='selu')(X)
    X = BatchNormalization()(X)
    return X

def convtranspose_bloc(X,filters,filter_size=(3,3),strides=2):
    X = Conv2DTranspose(filters,filter_size,strides=strides,activation='selu')(X)
    X = BatchNormalization()(X)
    return X

def make_encoder():
    encoder_input = keras.Input((height,width,3))

    X = conv_bloc(encoder_input,filters)
    for f in range(1,5):
        X = conv_bloc(X,filters*(2**f))
    X = conv_bloc(X,filters*32,filter_size=(3,3),strides=1)
    encoder_output = Conv2D(filters*64,(4,4),activation='selu')(X)

    encoder = keras.Model(inputs=encoder_input,
                         outputs=encoder_output)
    return encoder

def make_decoder(encoder):
    decoder_input = keras.Input(encoder.output.shape[1:])
    
    X = convtranspose_bloc(decoder_input,filters*64,(2,2),strides=1)
    for f in range(5):
        num_filters = filters*(2**(5-f))
        X = convtranspose_bloc(X,num_filters)
    X = convtranspose_bloc(X,filters,filter_size=(3,3))

    decoder_output = Conv2DTranspose(3,(2,2),activation='sigmoid')(X)

    decoder = keras.Model(inputs=decoder_input,
                         outputs=decoder_output)
    return decoder

ssim = lambda y_true,y_pred: 1 - tf.image.ssim(y_true,y_pred,max_val=1)
optimizer = keras.optimizers.Adam()

def make_autoencoder(encoder,decoder):
    autoencoder = keras.Sequential([
        encoder,
        decoder
    ])
    
    autoencoder.compile(loss=ssim,
                       optimizer=optimizer,
                       metrics='mae')
    return autoencoder


# ### Two models are created - one that cleans noisy images (called noise_autoencoder) and one to clean blurry images (blur_autoencoder). The noise autoencoder was trained first, with its weights then being transfered to the second model, which was then trained seperately.

# In[5]:


noise_encoder = make_encoder()
noise_decoder = make_decoder(noise_encoder)
noise_autoencoder = make_autoencoder(noise_encoder,
                                     noise_decoder)

blur_encoder = make_encoder()
blur_decoder = make_decoder(blur_encoder)
blur_autoencoder = make_autoencoder(blur_encoder,
                                   blur_decoder)

print('ENCODER:')
print(noise_encoder.summary())
print('\nDECODER:')
print(noise_decoder.summary())
print('\nAUTOENCODER:')
print(noise_autoencoder.summary())


# ### Each of the models was trained for 50+ hours on a GPU. The weights were downloaded and are uploaded here.

# In[6]:


noise_weights_path = os.path.join(cd,r'OneDrive\Desktop\Datasets\weights\image-autoencoder\noise-model-weights.h5')
noise_autoencoder.load_weights(noise_weights_path)

blur_weights_path = os.path.join(cd,r'OneDrive\Desktop\Datasets\weights\image-autoencoder\blur-model-weights.h5')
blur_autoencoder.load_weights(blur_weights_path)


# ### Here's how the models did on some of the images used for training (alternating between noisy and blurry images):

# In[7]:


print('Cleaning Training Images:')
for i in range(10):
    path = random.choice(train_paths)
    image = extract_and_resize_image(path)
    if i//2 == i/2:
        alterred_image = add_noise(image)
        cleaned_image = noise_autoencoder(tf.expand_dims(alterred_image,0)).numpy().squeeze()
    elif i//2 != 1/2:
        alterred_image = blur_image(image)
        cleaned_image = blur_autoencoder(tf.expand_dims(alterred_image,0)).numpy().squeeze()
    plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    plt.imshow(alterred_image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cleaned_image)
    plt.axis('off')
    plt.show()


# ### Here's how the model did on some of the test images:

# In[8]:


print('Cleaning Validation Images:')
for i in range(20):
    path = random.choice(validation_paths)
    image = extract_and_resize_image(path)
    if i//2 == i/2:
        alterred_image = add_noise(image)
        cleaned_image = noise_autoencoder(tf.expand_dims(alterred_image,0)).numpy().squeeze()
    elif i//2 != 1/2:
        alterred_image = blur_image(image)
        cleaned_image = blur_autoencoder(tf.expand_dims(alterred_image,0)).numpy().squeeze()
    plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    plt.imshow(alterred_image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cleaned_image)
    plt.axis('off')
    plt.show()

