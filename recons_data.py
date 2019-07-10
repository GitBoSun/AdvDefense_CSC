from __future__ import print_function
from builtins import input
from builtins import range
import pyfftw
import os
import numpy as np
import cv2
import sys
import sporco.metric as sm
from sporco.admm import cbpdn
from sporco import util
import multiprocessing
from multiprocessing import Pool

K = int(sys.argv[1]) # filter number, 64
PATCH_SIZE = int(sys.argv[2]) # filter size, 8
lmbda = float(sys.argv[3]) # sparse coefficient, 0.1 to 0.3 is good, we usually use 
IM_SIZE = int(sys.argv[4])
P = PATCH_SIZE
npd = 16
fltlmbd = 5
dic_path = 'basis_{}/{}_p{}_lm{}.npy'.format(IM_SIZE, K, P, lmbda)
# you need to specify the input path(im_path) and output path(out_path)
im_path = '224_res50/{}'.format(adv_type) 
out_path = '224_tmp'
basis = np.load(dic_path)
if not os.path.exists(out_path):
  os.makedirs(out_path)

def get_im(path):
  im = cv2.imread(path)
  im = cv2.resize(im, (IM_SIZE, IM_SIZE))
  im = np.array(im, np.float32)/255.0
  #im = np.expand_dims(im, -1)
  return im

def recons(name):
  if not os.path.exists(os.path.join(out_path, name)):
    impath = os.path.join(im_path, name)
    im = get_im(impath)
    sl, sh = util.tikhonov_filter(im, fltlmbd, npd)
    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 200, 'RelStopTol': 5e-3, 'AuxVarObj': False})
    b = cbpdn.ConvBPDN(basis, sh, lmbda, opt)
    X = b.solve()
    print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))
    shr = b.reconstruct().squeeze()
    imgr = sl + shr
    cv2.imwrite(os.path.join(out_path, name), 255*imgr/imgr.max())
  else:
#    print(name)
    pass  

if __name__=="__main__":
  names = sorted(os.listdir(im_path))  
  print(len(names))
  pool = Pool(1) #chage number of processings
  rl =pool.map(recons, names)
  pool.close()


