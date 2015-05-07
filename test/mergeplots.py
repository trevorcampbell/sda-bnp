import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'size' : 24}
matplotlib.rc('font', **font)



#create the plot output directory
sdabasename = 'sdadpmix'
batchbasename = 'vardpmix'
oldbatchbasename = 'vardpmixold'
svabasename = 'svadpmix'
svibasename = 'svidpmix'
movbbasename = 'movbdpmix'
outdir = 'plots'
if not os.path.exists(outdir):
    os.makedirs(outdir)

#get the mcmc run/nthr tags
mcmc_run_tags = []
nthr_tags = []
for f in os.listdir('.'):
    if f[:6] == 'model-':
        mcmc_run_tags.append(f[6:-4])
for f in os.listdir('.'):
    if f[:14] == sdabasename+'-nThr_' and f[-7:] == '-ab.log':
        for tag in mcmc_run_tags:
            if len(tag) < len(f[14:-7]) and f[-7-len(tag):-7] == tag:
                nthr_tags.append(f[14:-7-len(tag)-1])
mcmc_run_tags = sorted(mcmc_run_tags)
nthr_tags = sorted(list(set(nthr_tags)))


##Plot 1: data and model for each mcmc run
#for mtag in mcmc_run_tags:
#    model = np.genfromtxt('model-'+mtag+'.log')
#    train_data = np.genfromtxt('train-'+mtag+'.log')
#    test_data = np.genfromtxt('test-'+mtag+'.log')
#
#    D = int((-1 + np.sqrt(1-4*(1-model.shape[1])))/2)
#
#    plt.figure()
#    plt.scatter(train_data[:, 0], train_data[:, 1], c='b')
#    plt.scatter(test_data[:, 0], test_data[:, 1], c='g')
#    for k in range(model.shape[0]):
#        mu = model[k, 0:D]
#        sig = np.zeros((D, D))
#        for j in range(D):
#            sig[j, :] = model[k, (j+1)*D:(j+2)*D]
#        wt = model[k, -1]
#        xy  = mu[:, np.newaxis] + np.dot(np.linalg.cholesky(sig), np.vstack((np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))))
#        plt.plot(xy[0, :], xy[1, :], lw=5, alpha=np.sqrt(wt), c='r')
#    plt.axes().set_aspect('equal')
#    plt.xlim((-100, 100))
#    plt.ylim((-100, 100))
#    plt.savefig(outdir+'/model-'+mtag+'.pdf')



