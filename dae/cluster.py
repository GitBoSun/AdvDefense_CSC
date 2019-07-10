import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
batch_size = 32
num_classes = 10
epochs = 100
IM_SIZE = 64
train_path = '/data/imagenet/clean/dataset/train'
test_path = '/data/imagenet/clean/dataset/test'
saveDir = '/data/imagenet10/autoencoder/dae_keras/64'
model_path = os.path.join(saveDir, '64_AutoEncoder_denoise_weights.09-0.53-0.54.hdf5')

fea = np.load('imagenet10_test.npy')
test_fea = np.reshape(fea, [fea.shape[0], -1])
fea = np.load('imagenet10_train.npy')
train_fea = np.reshape(fea, [fea.shape[0], -1])
print(train_fea.shape, test_fea.shape)

val_fea = test_fea
#test_fea = test_fea[0:7000]
pca = PCA(n_components=128)
pca.fit(val_fea)
#val_pred = pca.transform(val_fea)
test_pred = pca.transform(test_fea)
train_pred = pca.transform(train_fea)
print('pca done', test_pred.shape, train_pred.shape)
total_fea = np.concatenate((train_pred, test_pred), axis=0)
print(total_fea.shape)
kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
kmeans.fit(total_fea)
Z = kmeans.predict(total_fea)
print(Z.shape)
l1 = train_pred.shape[0]
l2 = test_pred.shape[0]
train_names = sorted(os.listdir(train_path))
test_names = sorted(os.listdir(test_path))
fout = open('name_cluster_train.txt', 'w')
for i in range(l1):
  fout.write('%s %s\n'%(train_names[i], Z[i]))
fout.close()
fout = open('name_cluster_test.txt', 'w')
for i in range(l2):
  fout.write('%s %s\n'%(test_names[i], Z[i+l1]))
fout.close()
#tsne_fea = TSNE(n_components=2).fit_transform(val_pred)
#print('tsne done', tsne_fea.shape)
#np.savez('pca_whole_cifar.npz', tsne_fea = tsne_fea, pca_val=val_pred, pca_test=test_pred, pca_train=train_pred)



