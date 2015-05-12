import numpy as np
from scipy.linalg import eigh,inv
import os
import os.path
import cv2,ipdb
from config import Config2String
import matplotlib.pyplot as plt

def meanGaussian(eta,D):
  #ipdb.set_trace()
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
cfg['w'] = w = 64
cfg['h'] = h = 64
cfg['tag'] = 'sun397' # 'sun2012' # 'sun397'

# compute most likely indices
cfg['T'] = T = 10
cfg['Dout'] = Dout = 20

rootPath = '/data/vision/fisher/data1/sun2012/'
path2model = "./results/mergeout-K0-50-K_990-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.log"
path2model = "./results/sun/mergeoutFinal_A_8-K_75-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.log"
path2model = "./results/sun/mergeoutFinal_A_8-K_100-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.log" # starting to look good
path2model = "./results/sun/mergeoutFinal_A_8-K_300-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.log" 
path2model = "./results/sun/mergeoutFinal_A_8-K_125-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.log" 
path2model = "./sunDataHOGout.log" 
#new 20 dim ones for julian are below
#path2model = "./results/sun/mergeoutFinal_A_16-K_75-sunDataProjectedData-h_32-tag_sun397-Dout_20-w_32-N_108755.log" 
#path2model = "./results/sun/mergeoutFinal_A_-1-K_50-sunDataProjectedData-h_32-tag_sun397-Dout_20-w_32-N_108755.log" 

# run inference with sth like
#./build/vbInferCpu  -d ./data/sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.csv -p ./misc/prior50.csv -K 100 -A 8 -t A_8-K_100-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755  
# ./build/vbInferCpu  -d ./data/sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.csv -p ./misc/prior50.csv -K 75 -A 8 -t A_8-K_75-sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755   -o ./results/


if cfg['tag'] == 'sun2012':
  # SUN 2012
  # download http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz and extract then point to it:
  sunDataPath = '/data/vision/fisher/data1/sun2012/SUN2012/Images/'
  fin = open(rootPath+'sunDataPaths-h_32-tag_sun2012-Dout_'+str(Dout)+'-w_32-N_16874.txt')
  fileNames = fin.readlines()
  fileNames = [fName[:-1] for fName in fileNames]
  fin.close()
  x = np.loadtxt(rootPath+'sunDataProjectedData-h_32-tag_sun2012-Dout_'+str(Dout)+'-w_32-N_16874.csv')
else:
  # SUN 397
  # download http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz and extract then point to it:
  sunDataPath = '/data/vision/fisher/data1/sun2012/SUN397/'
  fin = open('./sunDataPaths.txt') #rootPath+'sunDataPaths-h_32-tag_sun397-Dout_50-w_32-N_108755.txt')
  fileNames = fin.readlines()
  fileNames = [fName[:-1] for fName in fileNames]
  fin.close()
  x = np.loadtxt('./sunDataHOGProjectedData.csv') #rootPath+'sunDataProjectedData-h_32-tag_sun397-Dout_50-w_32-N_108755.csv')
  #new stuff for julian below
  #if Dout == 20:
  #  fin = open(rootPath+'sunDataPaths-h_32-tag_sun397-Dout_'+str(Dout)+'-w_32-N_108755.txt')
  #  fileNames = fin.readlines()
  #  fileNames = [fName[:-1] for fName in fileNames]
  #  fin.close()
  #  fin = open(rootPath+'sunDataProjectedData-h_32-tag_sun397-Dout_'+str(Dout)+'-w_32-N_108755.csv')
  #  fin.readline()
  #  x = np.loadtxt(fin)
  #  fin.close()
  #elif Dout == 50:
  #  fin = open(rootPath+'sunDataPaths-h_32-tag_sun397-Dout_'+str(Dout)+'-w_32-N_108755.txt')
  #  fileNames = fin.readlines()
  #  fileNames = [fName[:-1] for fName in fileNames]
  #  fin.close()
  #  fin = open(rootPath+'sunDataProjectedData-h_32-tag_sun397-Dout_'+str(Dout)+'-w_32-N_108755.csv')
  #  fin.readline()
  #  x = np.loadtxt(fin)
  #  fin.close()

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
  print " loading inds from "+indPath
  ind = np.loadtxt(indPath).astype(np.int)
  Ks = ind[0,:]
  ind = ind[1::,:]
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
  
  Ks = np.argsort(pi)[-91:-1] # only look at biggest 100
  Ks = np.arange(pi.size)
  Ks = np.argsort(pi)[::-1] # sort from biggest to lowest
  print Ks
  print pi[Ks]
  # compute the T most likely data-points for each cluster
  ind = np.arange(T*Ks.size).reshape((T,Ks.size)) # TODO get this from actual algorithm
  pi = np.log(pi)
  for kk,k in enumerate(Ks):
    logPdf = np.zeros(N)
    for i in range(N):
      logPdf[i] = pi[k]+gaussianLogPdf(x[i,:],mus[k],invSs[k],logDets[k])
    indSorted = np.argsort(logPdf)[::-1]
    ind[:,kk] = indSorted[0:T];
    print '@k={}: {}'.format(k,ind[:,kk])
    print '  most likely: {} least likely: {}'.format(logPdf[indSorted[0:T]],logPdf[indSorted[-T-1:-1]])
  print Ks.shape
  print ind.shape
#  ipdb.set_trace() 
  ind = np.r_[Ks[np.newaxis,:], ind]
  np.savetxt(indPath,ind,fmt="%d")

# using those indices load the respective images and plot them intoa a single image
#ks = np.arange(K)[nus>100]
#ks = np.arange(70,130)
#ks = np.argsort(pi)[-30:-1]
#ks = np.argsort(nus)[-30:-1]
print Ks
for k0 in range(0,Ks.size,30):
  ks = np.arange(k0,k0+30) # Ks[np.arange(k0,k0+30)]
  K = ks.size
  Is = np.zeros((T*h,K*w,3),dtype=np.uint8)
  for j,k in enumerate(ks):
    for t in range(T):
      I = cv2.imread(fileNames[ind[t,k]])
      print fileNames[ind[t,k]], I.shape
      if not I is None and I.shape[0] >0 and I.shape[1] > 0 and I.shape[2] ==3:
        Is[t*h:(t+1)*h,j*w:(j+1)*w,:] = cv2.resize(I,(w,h))
#  fig = plt.figure()
#  tmp = Is[:,:,0]
#  Is[:,:,0] = Is[:,:,2]
#  Is[:,:,2] = tmp
#  plt.imshow(Is)
#  fig.show()
#  ipdb.set_trace()
  cv2.imwrite('classes_'+Config2String(cfg).toString()+'{}.jpg'.format(k0),Is)
#plt.show()
  cv2.imshow('image',Is)
  cv2.waitKey(0)
cv2.destroyAllWindows()


