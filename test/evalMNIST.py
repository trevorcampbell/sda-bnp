import numpy as np
from scipy.linalg import eigh,inv
import os
import os.path
import cv2
from config import Config2String
import matplotlib.pyplot as plt

def meanGaussian(eta,D):
  e1 = eta[0:D*D].reshape(D,D)
  e2 = eta[D*D:D*D+D]
  e3 = eta[D*D+D]
  nu = eta[D*D+D+1]
  # copute mean gaussian from those
  kappa = nu
  xi = e3 - D - 2.0
  th = e2/kappa
  psi = e1-kappa*th*th[:, np.newaxis]

  meancov = psi/(xi+D+1.0)
  meanmn = th
  return meanmn, meancov

def gaussianLogPdf(x,mu,S):
  D = mu.size
  logDet = np.linalg.slogdet(S)[1]
#  ipdb.set_trace()
  return -D/2.* 1.8378770664093453 - .5*logDet \
         - 0.5*(x-mu).dot(np.linalg.solve(S,x-mu))

def gaussianLogPdf(x,mu,invS,logDet):
#  ipdb.set_trace()
#  return -0.5*(D*np.log(2.*np.pi) +logDet \
  return -0.5*(mu.size* 1.8378770664093453 +logDet \
         +(x-mu).dot(invS.dot(x-mu)))

cfg=dict()
# compute most likely indices
cfg['T'] = T = 10
cfg['w'] = w = 28
cfg['h'] = h = 28

rootPath = './mnist/'
path2model = './mnist/mergeoutFinal_mnist-mergeout.log'
path2model = './mnist/mergeoutFinal_A_-1-K_20-MNIST.log'
path2model = './mnist/mergeoutFinal.log'

# run inference with sth like
# ./build/vbInferCpu  -d ./python/mnist/mnistProjectedData.csv -p ./misc/prior50.csv -K 20 -A -1 -t A_-1-K_20-MNIST   -o ./results/

fin = open(rootPath+'mnistProjectedData.csv')
fin.readline()
x = np.loadtxt(fin)
fin.close()
fileNames = [rootPath+'png/'+str(i)+'.png' for i in range(x.shape[0])]
N = len(fileNames)
print 'N = {} =? {}'.format(N,x.shape[0])
assert(N == x.shape[0])
cfg['N'] = N
cfg['D'] = D = x.shape[1]

# load the model
fin = open(path2model)
fin.readline() # ignore first line
eta = np.loadtxt(fin)
K = eta.shape[0];
nus = eta[:,-1]
print K
print eta.shape
print (nus > 100).sum()
#ipdb.set_trace()

indPath = path2model+'.ind-T_{}.csv'.format(T)

if os.path.isfile(indPath):
  ind = np.loadtxt(indPath).astype(np.int)
else:
  print eta.shape
  # compute the mean gaussian parameters of the mixture
  mus = []
  sigmas = []
  invSs = []
  logDets = []
  pi = np.log(np.ones(K)*(1./K))
  stickRemaining = 1.
  for k in range(0, K):
    mu,sigma = meanGaussian(eta[k,:],D)
    mus.append(mu)
    sigmas.append(sigma)

    logDets.append(np.linalg.slogdet(sigma)[1])
    invSs.append(inv(sigma))
    # TODO needs the newest type of merge parameter outputs
    a = eta[k,D*D+D+2]
    b = eta[k,D*D+D+3]
    stick = a/(a+b)
    pi[k] = stick*stickRemaining
    stickRemaining = (1. - stick)*stickRemaining

  print pi
  print pi.sum()
  Ks = np.argsort(pi)[::-1] # sort from biggest to lowest

  # compute the T most likely data-points for each cluster
  ind = np.arange(T*K).reshape((T,K)) # TODO get this from actual algorithm
  pi = np.log(pi)
  for kk,k in enumerate(Ks):
    logPdf = np.zeros(N)
    for i in range(N):
      logPdf[i] = pi[k]+gaussianLogPdf(x[i,:],mus[k],invSs[k],logDets[k])
    indSorted = np.argsort(logPdf)[::-1]
    ind[:,kk] = indSorted[0:T];
    print '@k={}: {}'.format(k,ind[:,kk])
    print '  most likely: {} least likely: {}'.format(logPdf[indSorted[0:T]],logPdf[indSorted[-T-1:-1]])

  np.savetxt(indPath,ind,fmt="%d")

# using those indices load the respective images and plot them intoa a single image
#ks = np.arange(K)[nus>100]
#ks = np.arange(70,130)
#ks = np.argsort(pi)[-30:-1]
#ks = np.argsort(nus)[-30:-1]
for k0 in range(0,K,26):
  ks = np.arange(k0,k0+26)
  K = ks.size
  Is = np.zeros((T*h,K*w),dtype=np.uint8)
  for j,k in enumerate(ks):
    for t in range(T):
      print k, ind[t,k]
      I = cv2.imread(fileNames[ind[t,k]], 0)
      print fileNames[ind[t,k]], I.shape
      if not I is None and I.shape[0] >0 and I.shape[1] > 0:
        Is[t*h:(t+1)*h,j*w:(j+1)*w] = cv2.resize(I,(w,h))
  cv2.imwrite(rootPath+'./mnistClustering_'+Config2String(cfg).toString()+'{}.jpg'.format(k0),Is)
  cv2.imshow('image',Is)
  cv2.waitKey(0)
  print "waiting "+'./classes_'+Config2String(cfg).toString()+'{}.jpg'

cv2.destroyAllWindows()


