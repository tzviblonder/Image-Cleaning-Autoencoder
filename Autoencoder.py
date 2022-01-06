#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To view the notebook with training, graphs, and images, go to: 
# https://nbviewer.org/github/tzviblonder/Image-Cleaning-Autoencoder/blob/main/autoencoder-to-denoise-images.ipynb

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

directory = '../input/celeba-dataset'

image_gen = ImageDataGenerator(rescale=1/255,
                              validation_split=.01)

height = 256
width = 256
batch_size = 64

training_images = image_gen.flow_from_directory(directory,
                                               class_mode=None,
                                               target_size=(height,width),
                                               batch_size=batch_size,
                                               subset='training')

validation_images = image_gen.flow_from_directory(directory,
                                                 class_mode=None,
                                                 target_size=(height,width),
                                                 batch_size=batch_size,
                                                 subset='validation')

def add_noise(image,height=height,width=width):
    noise = tf.random.normal(shape=(height,width,3),
                             mean=1,
                             stddev=1.)/5
    noisy_image = tf.add(image,noise)
    noisy_image = tf.clip_by_value(noisy_image,
                         clip_value_min=0,
                         clip_value_max=1)
    return noisy_image

def make_dataset(iterator,batch_size=batch_size,height=height,width=width,dtype=tf.float32):
    
    dataset = tf.data.Dataset.from_generator(lambda: (batch for batch in iterator),
                                            output_signature=tf.TensorSpec(shape=(None,height,width,3),
                                                                          dtype=dtype))
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size,
                           drop_remainder=True)
    dataset = dataset.map(lambda batch: (add_noise(batch),batch))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def random_flip(x,y):
    if random.randint(0,1) == 1:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    return x,y

train_dataset = make_dataset(training_images).map(random_flip)
validation_dataset = make_dataset(validation_images)

print(' '*15+'Clean Images:'+' '*40+'Noisy Images:')
for i in range(10):
    batch = next(iter(train_dataset))
    num = random.randint(0,len(batch)-1)
    clean_image = tf.cast(batch[1][num],tf.float64)
    noisy_image = tf.cast(batch[0][num],tf.float64)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(clean_image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(noisy_image)
    plt.axis('off')
    plt.show()
    
filters = 32

encoder = keras.Sequential([
    keras.layers.Conv2D(filters,(3,3),activation='elu',padding='same',input_shape=(height,width,3),kernel_regularizer='l2'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*2,(3,3),strides=2,kernel_regularizer='l2'),
    keras.layers.BatchNormalization(),
    keras.layers.ELU(),
    keras.layers.Conv2D(filters*4,(3,3),activation='elu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters*8,(3,3),strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ELU(),
    keras.layers.Conv2D(filters*16,(3,3),strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ELU(),
    keras.layers.Conv2D(filters*32,(3,3),activation='elu'),
    keras.layers.Conv2D(filters*32,(2,2),activation='elu')
])

encoder.summary()

decoder = keras.Sequential([
    keras.layers.Conv2DTranspose(filters*32,(3,3),input_shape=encoder.output.shape[1:]),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(.2),
    keras.layers.Conv2DTranspose(filters*32,(2,2),activation='relu',kernel_regularizer='l2'),
    keras.layers.UpSampling2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(filters*16,(3,3),activation='relu'),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*8,(3,3),activation='relu'),
    keras.layers.UpSampling2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(filters*4,(3,3),activation='relu'),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters*2,(3,3),activation='relu'),
    keras.layers.UpSampling2D(2),
    keras.layers.Conv2DTranspose(filters,(3,3),activation='relu'),
    keras.layers.Conv2DTranspose(3,(3,3),activation='sigmoid')
])

decoder.summary()

autoencoder = keras.Model(inputs=encoder.inputs,
                         outputs=decoder(encoder.outputs))

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

def ssim(x,y):
    return 1 - tf.image.ssim(x,y,max_val=1)

autoencoder.compile(loss=ssim,
                   optimizer=optimizer,
                   metrics='mae')

autoencoder.summary()

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             patience=2,
                                             verbose=1,
                                             min_lr=1e-7,
                                             min_delta=.001)

early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                          patience=4,
                                          verbose=1,
                                          min_delta=.001,
                                          restore_best_weights=True)

epochs = 18
steps_per_epoch = len(training_images)
validation_steps = len(validation_images)

history = autoencoder.fit(train_dataset,
                         validation_data=validation_dataset,
                         epochs=epochs,
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         callbacks=[reduce_lr,
                                   early_stop])

hist = history.history
loss = hist['loss']
mae = hist['mae']
val_loss = hist['val_loss']
val_mae = hist['val_mae']
epoch = np.arange(len(loss)) + 1

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

print('Denoising training images:')
for i in range(25):
    batch_num = random.randint(0,len(training_images)-1)
    batch = training_images[batch_num]
    img_num = random.randint(0,len(batch)-1)
    image = batch[img_num]
    noisy_image = add_noise(image)
    denoised_image = autoencoder(tf.expand_dims(noisy_image,axis=0)).numpy().squeeze()
    plt.figure(figsize=(12,6))
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
    
print('Denoising validation images:')
for i in range(25):
    batch_num = random.randint(0,len(validation_images)-1)
    batch = validation_images[batch_num]
    img_num = random.randint(0,len(batch)-1)
    image = batch[img_num]
    noisy_image = add_noise(image)
    denoised_image = autoencoder(tf.expand_dims(noisy_image,axis=0)).numpy().squeeze()
    plt.figure(figsize=(12,6))
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

