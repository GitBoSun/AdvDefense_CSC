import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
print(kerasBKED)
import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import cv2
import pickle
import numpy as np
from keras import backend as K
batch_size = 32
num_classes = 10
IM_SIZE = 64
epochs = 100
saveDir = '/data/ImageNet/autoencoder/dae_keras'
model_path = os.path.join(saveDir, '6464_AutoEncoder_denoise_weights.15-0.53-0.53.hdf5')
data_path = '/data/ImageNet/val'

cls_map_txt = '/home/bosun/imagenet/utils/map_cls.txt'
cls_txt = '/home/bosun/retrieval/tf_vgg/tensorflow-vgg/synset.txt'
def get_cls_map():
  cls_map = {}
  f = open(cls_txt, "r")
  for i, line in enumerate(f.readlines()):
#    cls, _ = line.split(' ')
    cls = line[:9]
    cls_map[cls] = i
  f.close()
  return cls_map

def get_data(path):
  test_data = []
  test_label = []
  labels = sorted([label for label in os.listdir(path)])
  cls_map = get_cls_map()
  for i, label in enumerate(labels):
    j=0
    for name in os.listdir(os.path.join(path, label)):
        j+=1
      #try:
        if j<6:
          continue
        im = cv2.imread(os.path.join(path, label, name))
        im = cv2.resize(im, (IM_SIZE, IM_SIZE))
        test_data.append(im)
        y = int(cls_map[label])
        test_label.append(y)
        #j+=1
        if j==11:
          break
      #except:
      #  print(name)
      #  continue
  test_data = np.array(test_data, np.float32)
  test_label = np.array(test_label, np.float32)
  return test_data/255.0, test_label


x_test, y_test = get_data(data_path)
print(x_test.shape[0], 'test samples')

noise_factor = 0.15
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(IM_SIZE, IM_SIZE, 3))
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
model.load_weights(model_path)
print(model.layers[11].output)
inp = model.input
feature = [model.layers[12].output]
functor = K.function([inp, K.learning_phase()], feature)
layer_outs = functor([x_test_noisy, 1.])
outs = np.array(layer_outs)
print(outs.shape)
np.savez('imagenet_test_2.npz', fea=outs[0], im = x_test, label=y_test)
#es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
#chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
#cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#score = model.evaluate(x_test_noisy, x_test, verbose=1)
#print(score)
                                                                     

