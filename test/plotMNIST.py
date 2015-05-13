import numpy as np
from scipy.linalg import inv
import os
import os.path
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

sns.set(font_scale=1.5)
snsm = sns.color_palette('muted')
rc('text', usetex=False)

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

def gaussianLogPdf(x,mu,invS,logDet):
  return -0.5*(mu.size*np.log(2*np.pi) +logDet \
         +(x-mu).dot(invS.dot(x-mu)))

tracefiles = sorted([fname for fname in os.listdir('.') if fname[-9:]=='trace.log' ])
etafiles = sorted([fname for fname in os.listdir('.') if fname[-7:]=='eta.log' ])
abfiles = sorted([fname for fname in os.listdir('.') if fname[-6:]=='ab.log' ])
nufiles = sorted([fname for fname in os.listdir('.') if fname[-6:]=='nu.log' ])
train_data = np.loadtxt('mnistTrain20.txt')
imgFileNames = ['./mnistimgs/' + str(i)+'.png' for i in range(train_data.shape[0])]
imgOutRoot = 'mnistclasses'

nClusToShow = 15
nImgPerClus = 15
imgW = 28
imgH = 28

##TODO -- possibly filter out small clusters before plotting

#lglbls = []
#plt.figure()
#for i in range(len(tracefiles)):
#    tr = np.genfromtxt(tracefiles[i])
#    times = tr[:, 0]
#    testlls = tr[:, 1]
#    plt.plot(times, testlls, c=snsm[i%len(snsm)], lw=2)
#    lglbls.append(tracefiles[i][:-10])
#plt.legend(lglbls)
#plt.xscale('log')
#
#plt.show()
#quit()

for i in range(len(etafiles)):
    outname = etafiles[i][:-8]
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
      invSs.append(inv(sigma))
      a = ab[0, k]
      b = ab[1, k]
      stick = a/(a+b)
      pi[k] = stick*stickRemaining
      stickRemaining = (1. - stick)*stickRemaining
    pi[K-1] = 1.0-np.sum(pi[:K-1])

    # compute the T most likely data-points for each cluster
    ind = np.zeros((train_data.shape[0], K))
    for k in range(K):
        logPdf = np.zeros(train_data.shape[0])
        for j in range(train_data.shape[0]):
            logPdf[j] = np.log(pi[k])+gaussianLogPdf(train_data[j,:],mus[k],invSs[k],logDets[k])
        ind[:, k] = np.argsort(logPdf)[::-1]
    ind = ind.astype(int)

    # using those indices load the respective images and plot them intoa a single image
    #ks = np.arange(K)[nus>100]
    #ks = np.arange(70,130)
    #ks = np.argsort(pi)[-30:-1]
    #ks = np.argsort(nus)[-30:-1]
    for k0 in range(0, K, nClusToShow):
        ks = np.arange(k0, min(k0+nClusToShow, K))
        Is = np.zeros((nImgPerClus*imgH, ks.shape[0]*imgW))
        for j,k in enumerate(ks):
            for t in range(nImgPerClus):
                print k, ind[t,k]
                I = cv2.imread(imgFileNames[ind[t,k]], 0)
                print imgFileNames[ind[t,k]], I.shape
                if not I is None and I.shape[0] >0 and I.shape[1] > 0:
                    Is[t*imgH:(t+1)*imgH,j*imgW:(j+1)*imgW] = cv2.resize(I,(imgW,imgH))
        cv2.imwrite(imgOutRoot+'/mnist_'+outname+'{}.jpg'.format(k0),Is)
        #cv2.imshow('image',Is)
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()


