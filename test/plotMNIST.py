import numpy as np
from scipy.linalg import eigh,inv
import os
import os.path
import cv2
from config import Config2String
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5)
snsm = sns.color_palette('muted')

def MtoD(M):
    return int((-1+np.sqrt(-3+4*M))/2)

def meanGaussian(eta, nu, D):
  e1 = eta[0:D*D].reshape(D,D)
  e2 = eta[D*D:D*D+D]
  e3 = eta[D*D+D]
  # copute mean gaussian from those
  kappa = nu
  xi = e3 - D - 2.0
  th = e2/kappa
  psi = e1-kappa*th*th[:, np.newaxis]

  meancov = psi/(xi+D+1.0)
  meanmn = th
  return meanmn, meancov

#def gaussianLogPdf(x,mu,S):
#  D = mu.size
#  logDet = np.linalg.slogdet(S)[1]
##  ipdb.set_trace()
#  return -D/2.* 1.8378770664093453 - .5*logDet \
#         - 0.5*(x-mu).dot(np.linalg.solve(S,x-mu))

def gaussianLogPdf(x,mu,invS,logDet):
#  ipdb.set_trace()
#  return -0.5*(D*np.log(2.*np.pi) +logDet \
  return -0.5*(mu.size* 1.8378770664093453 +logDet \
         +(x-mu).dot(invS.dot(x-mu)))

tracefiles = sorted([fname for fname in os.listdir() if fname[-9:]=='trace.log' ])
etafiles = sorted([fname for fname in os.listdir() if fname[-7:]=='eta.log' ])
abfiles = sorted([fname for fname in os.listdir() if fname[-6:]=='ab.log' ])
nufiles = sorted([fname for fname in os.listdir() if fname[-6:]=='nu.log' ])
train_data = np.loadtxt('mnistTrain20.txt')
imgFileNames = ['imgs/'+str(i)+'.png' for i in range(train_data.shape[0])]

lglbls = []
plt.figure()
for i in range(len(tracefiles)):
    tr = np.genfromtxt(tracefiles[i])
    times = tr[:, 0]
    testlls = tr[:, 1]
    plt.plot(times, testlls, c=snsm[i], lw=2)
    lglbls.append(tracefiles[i])
plt.legend(lglbls)
plt.xscale('log')

for i in range(len(etafiles)):
    eta = np.genfromtxt(etafiles[i])
    ab = np.genfromtxt(abfiles[i])
    nu = np.genfromtxt(nufiles[i])
    K = eta.shape[0]
    D = MtoD(eta.shape[1])
    # compute the mean gaussian parameters of the mixture
    mus = []
    sigmas = []
    invSs = []
    logDets = []
    pi = np.zeros(K)
    stickRemaining = 1.
    for k in range(K):
      mu,sigma = meanGaussian(eta[k,:], nu[k], D)
      mus.append(mu)
      sigmas.append(sigma)

      logDets.append(np.linalg.slogdet(sigma)[1])
      invSs.append(np.linalg.inv(sigma))
      a = ab[0, k]
      b = ab[1, k]
      stick = a/(a+b)
      pi[k] = stick*stickRemaining
      stickRemaining = (1. - stick)*stickRemaining
    pi[K-1] = 1.0-np.sum(pi[:K-1])

    # compute the T most likely data-points for each cluster
    ind = np.zeros((N, K))
    for k in range(K):
        logPdf = np.zeros(N)
        for j in range(N):
            logPdf[j] = np.log(pi[k])+gaussianLogPdf(train_data[j,:],mus[k],invSs[k],logDets[k])
        ind[:, k] = np.argsort(logPdf)[::-1]

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


