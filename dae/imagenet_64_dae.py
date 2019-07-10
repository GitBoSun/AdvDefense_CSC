import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(kerasBKED)
import keras
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import cv2
from data_gen import DataGenerator

batch_size = 32
num_classes = 10
im_size = 64
num_epochs = 100
train_path = '/data/ImageNet/data64/train'
val_path = '/data/ImageNet/data64/val'
saveDir = '/data/ImageNet/autoencoder/dae_keras/64' 

if not os.path.isdir(saveDir):
  os.makedirs(saveDir)

training_generator = DataGenerator(train_path)
validation_generator = DataGenerator(val_path)

input_img = Input(shape=(im_size, im_size, 3))
x = Conv2D(32, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

model = Model(input_img, decoded)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')

es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
chkpt = saveDir + '64_AutoEncoder_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=40000,
                    epochs=num_epochs,
                    verbose=1,
                    validation_steps=200,
                    callbacks=[es_cb, cp_cb],
                    use_multiprocessing=True,
                    workers=1)

