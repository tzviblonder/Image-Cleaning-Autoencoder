#!/usr/bin/env python
# coding: utf-8

# ### Camera images are often unclear do to factors such as lighting, movement, and the age or quality of the camera, all of which can add "noise" to an image. This algorithm attempts to extract a clear image from a noisy one.

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os

directory = '../input/celeba-dataset/img_align_celeba/img_align_celeba'
image_paths = []

for image_path in os.listdir(directory):
    image_paths.append(os.path.join(directory,image_path))

random.shuffle(image_paths)

test_size = 4000
train_paths = image_paths[:-test_size]
validation_paths = image_paths[-test_size:]


# In[2]:


height = 256
width = 256
noise_param = 7
batch_size = 80

def extract_and_resize_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image,tf.float16)
    image.set_shape(image.shape)
    image = tf.image.resize(image,(height,width))
    return image

def add_noise(image,noise_param=noise_param):
    noise = tf.random.normal(shape=(height,width,3),
                            mean=1,
                            stddev=1.)/noise_param
    noisy_image = tf.add(image,noise)
    noisy_image = tf.clip_by_value(noisy_image,
                                  clip_value_min=0,
                                  clip_value_max=1)
    return noisy_image

def augment(noisy_image,image):
    if random.randint(0,1) == 1:
        noisy_image = tf.image.flip_left_right(noisy_image)
        image = tf.image.flip_left_right(image)
    if random.randint(0,3) == 1:
        noisy_image = tf.image.flip_up_down(noisy_image)
        image = tf.image.flip_up_down(image)
    return noisy_image,image

def make_dataset(paths,train=True,batch_size=batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(extract_and_resize_image)
    dataset = dataset.map(lambda image: (add_noise(image),image))
    if train:
        dataset = dataset.map(augment)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = make_dataset(train_paths)
validation_dataset = make_dataset(validation_paths,train=False)


# In[3]:


print(' '*15+'Clean Images:'+' '*40+'Noisy Images:')
for _ in range(12):
    path = random.choice(train_paths)
    image = extract_and_resize_image(path)
    image = tf.cast(image,tf.float32)
    noisy_image = add_noise(image)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(noisy_image)
    plt.axis('off')
    plt.show()


# ### This model is an autoencoder, which uses an encoder model to condense the images into a smaller-dimensional space. Then, a decoder uses the encoded image to rebuild it without the noise.

# In[4]:


filters = 24
l2 = keras.regularizers.L2(1e-6)

encoder = keras.Sequential([
    keras.layers.Conv2D(filters,(3,3),activation='elu',padding='same',
                        input_shape=(height,width,3),kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*2,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*4,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*8,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*16,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*32,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters*64,(3,3),activation='elu',kernel_regularizer=l2)
])

encoder.summary()


# In[5]:


decoder = keras.Sequential([
    keras.layers.Conv2DTranspose(filters*64,(3,3),input_shape=encoder.output.shape[1:],kernel_regularizer=l2),
    keras.layers.LeakyReLU(.1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(filters*32,(3,3),kernel_regularizer=l2),
    keras.layers.LeakyReLU(.1),
    keras.layers.BatchNormalization(),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*16,(3,3),activation='elu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*8,(3,3),activation='relu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*4,(3,3),activation='relu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*2,(3,3),activation='relu',kernel_regularizer=l2),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters,(3,3),activation='relu',kernel_regularizer=l2),
    keras.layers.Conv2DTranspose(3,(3,3),activation='sigmoid')
])

decoder.summary()


# In[6]:


autoencoder = keras.Model(inputs=encoder.inputs,
                         outputs=decoder(encoder.outputs))

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

def ssim(x,y):
    return 1 - tf.image.ssim(x,y,max_val=1)

autoencoder.compile(loss=ssim,
                   optimizer=optimizer,
                   metrics='mae')

autoencoder.summary()


# In[7]:


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             patience=2,
                                             verbose=1,
                                             min_lr=1e-5,
                                             min_delta=5e-3)

epochs = 24

history = autoencoder.fit(train_dataset,
                         validation_data=validation_dataset,
                         epochs=epochs,
                         callbacks=[reduce_lr])


# In[8]:


hist = history.history
loss = hist['loss']
mae = hist['mae']
val_loss = hist['val_loss']
val_mae = hist['val_mae']
epoch = np.arange(epochs) + 1

sns.set_style('darkgrid')
plt.figure(figsize=(12,7))
plt.plot(epoch,loss)
plt.plot(epoch,val_loss)
plt.title('Training & Validation Loss (Mean Squared Error)',fontdict={'fontsize':20})
plt.xlabel('Epoch',fontdict={'fontsize':16})
plt.ylabel('Loss',fontdict={'fontsize':16})
plt.legend(['Training Loss','Validation Loss'],prop={'size':18})
plt.show()

plt.figure(figsize=(12,7))
plt.plot(epoch,mae)
plt.plot(epoch,val_mae)
plt.title('Training & Validation Mean Absolute Error',fontdict={'fontsize':20})
plt.xlabel('Epoch',fontdict={'fontsize':16})
plt.ylabel('Mean Absolute Error',fontdict={'fontsize':16})
plt.legend(['Training MAE','Validation MAE'],prop={'size':18})
plt.show()


# In[9]:


print('Denoising training images:')
for i in range(25):
    path = random.choice(train_paths)
    image = extract_and_resize_image(path)
    noisy_image = add_noise(image)
    denoised_image = autoencoder(tf.expand_dims(noisy_image,0)).numpy().squeeze()
    plt.figure(figsize=(18,8))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(noisy_image)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(denoised_image)
    plt.axis('off')
    plt.show()


# In[10]:


print('Denoising validation images:')
for i in range(35):
    path = random.choice(validation_paths)
    image = extract_and_resize_image(path)
    noisy_image = add_noise(image)
    denoised_image = autoencoder(tf.expand_dims(noisy_image,0)).numpy().squeeze()
    plt.figure(figsize=(18,8))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(noisy_image)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(denoised_image)
    plt.axis('off')
    plt.show()

