import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set(font_scale=1.5)
snsm = sns.color_palette('muted')

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#font = {'family' : 'normal',
#        'size' : 24}
#matplotlib.rc('font', **font)



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

#THE BELOW LINE REMOVES LARGER THREAD NUMBERS
#IF YOU WANT THEM BACK JUST REMOVE THIS
##############
##########
nthr_tags = nthr_tags[:6]
##########
#############

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


#collect traces/final statistics/etc
final_cpu_times = np.zeros((len(nthr_tags), len(mcmc_run_tags)))
final_testlls = np.zeros((len(nthr_tags), len(mcmc_run_tags)))
final_merge_times = np.zeros((len(nthr_tags), len(mcmc_run_tags)))
final_nclus = np.zeros((len(nthr_tags), len(mcmc_run_tags)))
final_nmatch = np.zeros((len(nthr_tags), len(mcmc_run_tags)))
for i in range(len(nthr_tags)):
    ntag = nthr_tags[i]
    for j in range(len(mcmc_run_tags)):
        mtag = mcmc_run_tags[j]
        gt = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-trace.log')
        final_cpu_times[i, j] = gt[-1, 0]
        final_testlls[i, j] = gt[-1, 1]
        final_merge_times[i, j] = gt[-1, 3]
        final_nclus[i, j] = gt[-1, 4]
        final_nmatch[i, j] = gt[-1, 5]


#Plot 2: bar graph across all mcmc runs, x axis nThr, y axis model quality and cpu time with merge time stacked onto it
plt.figure()
axl = plt.axes()
axr = axl.twinx()
barwidth = .35
plt.sca(axl)
bp1 = axl.bar(-barwidth/2.0+np.arange(final_cpu_times.shape[0]), np.mean(final_cpu_times, axis=1), barwidth, color=snsm[0], lw=1.5, ec=(0.3, 0.3, 0.3), error_kw=dict(ecolor=(0.3, 0.3, 0.3), lw=1.5, capsize=5, capthick=2), yerr=np.std(final_cpu_times, axis=1))
#axr.errorbar(barwidth+np.arange(final_testlls.shape[0]), np.mean(final_testlls, axis=1), fmt='o', ms=10, color='r', yerr=np.std(final_testlls, axis=1))
#sns.boxplot(list(final_cpu_times), positions=np.arange(final_cpu_times.shape[0]), color='skyblue')
#bp1, = plt.bar(0, 0, 0, color='skyblue')
plt.sca(axr)
sns.boxplot(list(final_testlls), positions=np.arange(final_testlls.shape[0]), color=snsm[1], widths=.4)
bp2, = plt.bar(0, 0, 0, color=snsm[1], lw=1.5, ec=(0.3, 0.3, 0.3))
#axr.set_xticks(barwidth+np.arange(final_testlls.shape[0]))
axr.set_xticks(np.arange(final_testlls.shape[0]))
axr.set_xticklabels( map(str, map(int, nthr_tags)) )
axl.set_xlabel(r'\# Threads')
axl.set_ylabel(r'CPU Time (s)')
axr.set_ylabel(r'Test Log Likelihood')
axr.set_ylim((-12, -4))
axl.set_ylim((0, 1))
axr.set_yticks(np.linspace(axr.get_yticks()[0], axr.get_yticks()[-1], len(axl.get_yticks())))
axr.grid(None)
axr.set_yticklabels(axr.get_yticks().tolist()) #these two commands for some reason help formatting the y axis tick labels...
axl.set_yticklabels(axl.get_yticks().tolist())
plt.legend([bp1, bp2], ['CPU Time', 'Test Log Likelihood'])
plt.savefig(outdir+'/cput-testll-boxes.pdf')

#Plot 2: Merge times
plt.figure()
fmtf = final_merge_times.flatten()
#plt.hist(fmtf, bins=np.logspace(np.log10(np.amin(fmtf)), np.log10(np.amax(fmtf)), 30), facecolor='skyblue')
#plt.xscale('log')
plt.hist(fmtf*1.0e6, bins=np.linspace(np.amin(fmtf)*1.0e6, np.amax(fmtf)*1.0e6, 30), facecolor=snsm[0], ec=(0.3, 0.3, 0.3), lw=1.5)
plt.axes().set_yticklabels(map(int, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...
plt.axes().set_xticklabels(map(int, plt.axes().get_xticks().tolist()))
plt.xlabel(r'Merge Time (microseconds)')
plt.ylabel(r'Count')
plt.savefig(outdir+'/merget-hist.pdf')

#Plot 3: Trace of number of clusters/matchings vs number of merged minibatch posteriors for the max # threads
nclus_traces = []
nmatch_traces = []
ntag = nthr_tags[3]
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    gt = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-trace.log')
    nclus_traces.append(gt[:, 4])
    nmatch_traces.append(gt[:, 5])

plt.figure()
ctrace_mean = np.mean(np.array(nclus_traces), axis=0)
ctrace_std = np.std(np.array(nclus_traces), axis=0)
mtrace_mean = np.mean(np.array(nmatch_traces), axis=0)
mtrace_std = np.std(np.array(nmatch_traces), axis=0)

p1, = plt.plot( 1+np.arange(ctrace_mean.shape[0]), ctrace_mean, c=snsm[0], lw=2)
plt.plot( 1+np.arange(ctrace_mean.shape[0]), ctrace_mean+ctrace_std, c=snsm[0], ls='--', lw=2)
plt.plot( 1+np.arange(ctrace_mean.shape[0]), ctrace_mean-ctrace_std, c=snsm[0], ls='--', lw=2)
plt.fill_between( 1+np.arange(ctrace_mean.shape[0]), ctrace_mean-ctrace_std, ctrace_mean + ctrace_std, facecolor=snsm[0], alpha=0.3)

p2, =plt.plot( 1+np.arange(mtrace_mean.shape[0]), mtrace_mean, c=snsm[1], lw=2)
plt.plot( 1+np.arange(mtrace_mean.shape[0]), mtrace_mean+mtrace_std, c=snsm[1], ls='--', lw=2)
plt.plot( 1+np.arange(mtrace_mean.shape[0]), mtrace_mean-mtrace_std, c=snsm[1], ls='--', lw=2)
plt.fill_between( 1+np.arange(mtrace_mean.shape[0]), mtrace_mean-mtrace_std, mtrace_mean + mtrace_std, facecolor=snsm[1], alpha=0.3)
plt.ylim([0, 60])
plt.axes().set_yticklabels(map(int, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...)
plt.axes().set_xticklabels(map(int, plt.axes().get_xticks().tolist()))
plt.legend([p1, p2], [r'\# Clusters', r'\# Matchings'])
plt.xlabel(r'\# Minibatches Merged')
plt.ylabel(r'Count')
plt.savefig(outdir+'/nclusmatch-lines.pdf')




#Plot 3: Number of clusters (one line for each nThr) & number of matchings solved vs # merged minibatch posteriors
ff = plt.figure()
plt.bar(np.arange(final_nclus.shape[0]), np.mean(final_nclus, axis=1), barwidth, color=snsm[0], ec=(0.3, 0.3, 0.3), lw=1.5, error_kw=dict(ecolor=(0.3, 0.3, 0.3), lw=1.5, capsize=5, capthick=2), yerr=np.std(final_nclus, axis=1))
plt.bar(barwidth+np.arange(final_nmatch.shape[0]), np.mean(final_nmatch, axis=1), barwidth, color=snsm[1], ec=(0.3, 0.3, 0.3), lw=1.5, error_kw=dict(ecolor=(0.3, 0.3, 0.3), lw=1.5, capsize=5, capthick=2), yerr=np.std(final_nmatch, axis=1))
plt.xlabel(r'\# Threads')
plt.ylabel(r'Count')
plt.ylim([0, 60])
ax = plt.axes()
ax.set_xticks(barwidth+np.arange(final_nclus.shape[0]))
ax.set_xticklabels( map(str, map(int, nthr_tags)) )
plt.axes().set_yticklabels(map(int, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...)
plt.legend([r'\# Clusters', r'\# Matchings'])
plt.savefig(outdir+'/nclusmatch-bars.pdf')



####Plot 5: Global/minibatch likelihood traces for a single MCMC run
####use the 32thread / 1st mcmc run (totally fine to change these)
####ntag = nthr_tags[-1]
####mtag = mcmc_run_tags[-1]
####gt = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-globaltrace.log')
####global_times = gt[:, 0]
####global_testlls = gt[:, 1]
####lt = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-localtimes.log')
####local_start_times = lt[:, 0]
####local_testlls = []
####local_times = []
####for i in range(lt.shape[0]):
####    lt = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-localtrace-'+str(i)+'.log')
####    local_testlls.append(lt[:,2])
####    local_times.append(lt[:,0])
####
####plt.figure()
####plt.plot(global_times, global_testlls, 'b', lw=2)
####for  i in range(len(local_times)):
####    plt.plot(local_start_times[i]+local_times[i], local_testlls[i], 'c', lw=1)
####plt.xlabel('Time (s)')
####plt.ylabel('Test Log Likelihood')
####plt.xscale('log')
####plt.savefig(outdir+'/testll-trace.pdf')

plt.figure()
lglbls = []
lghdls = []
##Plot 6: Test log likelihood vs time with mean/std
#for i in range(len(nthr_tags)):
#    ntag = nthr_tags[i]
#    sda_times = []
#    sda_testlls = []
#    for j in range(len(mcmc_run_tags)):
#        mtag = mcmc_run_tags[j]
#        tr = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-trace.log')
#        sda_times.append(tr[:, 0])
#        sda_testlls.append(tr[:, 1])
#    minTime = np.amin(np.array(map(np.amin, sda_times)))
#    maxTime = np.amax(np.array(map(np.amax, sda_times)))
#    t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
#    tllp = np.zeros((len(sda_times), t.shape[0]))
#    for j in range(len(sda_times)):
#        tllp[j, :] = np.interp(t, sda_times[j], sda_testlls[j])
#    tll_mean = np.mean(tllp, axis=0)
#    tll_std = np.std(tllp, axis=0)
#    lghdl, = plt.plot(t, tll_mean, c=snsm[0], lw=2)
#    lghdls.append(lghdl)
#    plt.plot(t, tll_mean+tll_std, c=snsm[0], ls='--', lw=2, alpha=0.4)
#    plt.plot(t, tll_mean-tll_std, c=snsm[0], ls='--', lw=2, alpha=0.4)
#    plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[0], alpha=0.3)
#    lglbls.append('SDA-DP-'+ntag)

ntag = nthr_tags[3]
sda_times = []
sda_testlls = []
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    tr = np.genfromtxt(sdabasename+'-nThr_'+ntag+'-'+mtag+'-trace.log')
    sda_times.append(tr[:, 0])
    sda_testlls.append(tr[:, 1])
minTime = np.amin(np.array(map(np.amin, sda_times)))
maxTime = np.amax(np.array(map(np.amax, sda_times)))
t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
tllp = np.zeros((len(sda_times), t.shape[0]))
for j in range(len(sda_times)):
    tllp[j, :] = np.interp(t, sda_times[j], sda_testlls[j])
tll_mean = np.mean(tllp, axis=0)
tll_std = np.std(tllp, axis=0)
lghdl, = plt.plot(t, tll_mean, c=snsm[0], lw=2)
lghdls.append(lghdl)
plt.plot(t, tll_mean+tll_std, c=snsm[0], ls='--', lw=2, alpha=0.4)
plt.plot(t, tll_mean-tll_std, c=snsm[0], ls='--', lw=2, alpha=0.4)
plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[0], alpha=0.3)
lglbls.append('SDA-DP')


#batch_times =[]
#batch_testlls =[]
#for j in range(len(mcmc_run_tags)):
#    mtag = mcmc_run_tags[j]
#    tr = np.genfromtxt(batchbasename+'-'+mtag+'-trace.log')
#    batch_times.append(tr[:, 0])
#    batch_testlls.append(tr[:, 2]) #2 for this one since 1 is obj
#minTime = np.amin(np.array(map(np.amin, batch_times)))
#maxTime = np.amax(np.array(map(np.amax, batch_times)))
#t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
#tllp = np.zeros((len(batch_times), t.shape[0]))
#for j in range(len(batch_times)):
#    tllp[j, :] = np.interp(t, batch_times[j], batch_testlls[j])
#tll_mean = np.mean(tllp, axis=0)
#tll_std = np.std(tllp, axis=0)
#plt.plot(t, tll_mean, c=snsm[1], lw=2)
#plt.plot(t, tll_mean+tll_std, c=snsm[1], ls='--', lw=2, alpha=0.4)
#plt.plot(t, tll_mean-tll_std, c=snsm[1], ls='--', lw=2, alpha=0.4)
#plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[1], alpha=0.3)


oldbatch_times = []
oldbatch_testlls = []
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    tr = np.genfromtxt(oldbatchbasename+'-'+mtag+'-trace.log')
    oldbatch_times.append(tr[:, 0])
    oldbatch_testlls.append(tr[:, 1])
minTime = np.amin(np.array(map(np.amin, oldbatch_times)))
maxTime = np.amax(np.array(map(np.amax, oldbatch_times)))
t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
tllp = np.zeros((len(oldbatch_times), t.shape[0]))
for j in range(len(oldbatch_times)):
    tllp[j, :] = np.interp(t, oldbatch_times[j], oldbatch_testlls[j])
tll_mean = np.mean(tllp, axis=0)
tll_std = np.std(tllp, axis=0)
lghdl, = plt.plot(t, tll_mean, c=snsm[2], lw=2)
lghdls.append(lghdl)
plt.plot(t, tll_mean+tll_std, c=snsm[2], ls='--', lw=2, alpha=0.4)
plt.plot(t, tll_mean-tll_std, c=snsm[2], ls='--', lw=2, alpha=0.4)
plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[2], alpha=0.3)
lglbls.append('Batch')

sva_times = []
sva_testlls =[]
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    tr = np.genfromtxt(svabasename+'-'+mtag+'-trace.log')
    sva_times.append(tr[:, 0])
    sva_testlls.append(tr[:, 1])
minTime = np.amin(np.array(map(np.amin, sva_times)))
maxTime = np.amax(np.array(map(np.amax, sva_times)))
t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
tllp = np.zeros((len(sva_times), t.shape[0]))
for j in range(len(sva_times)):
    tllp[j, :] = np.interp(t, sva_times[j], sva_testlls[j])
tll_mean = np.mean(tllp, axis=0)
tll_std = np.std(tllp, axis=0)
lghdl, = plt.plot(t, tll_mean, c=snsm[3], lw=2)
lghdls.append(lghdl)
plt.plot(t, tll_mean+tll_std, c=snsm[3], ls='--', lw=2, alpha=0.4)
plt.plot(t, tll_mean-tll_std, c=snsm[3], ls='--', lw=2, alpha=0.4)
plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[3], alpha=0.3)
lglbls.append('SVA')

svi_times = []
svi_testlls = []
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    tr = np.genfromtxt(svibasename+'-'+mtag+'-trace.log')
    svi_times.append(tr[:, 0])
    svi_testlls.append(tr[:, 1])
minTime = np.amin(np.array(map(np.amin, svi_times)))
maxTime = np.amax(np.array(map(np.amax, svi_times)))
t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
tllp = np.zeros((len(svi_times), t.shape[0]))
for j in range(len(svi_times)):
    tllp[j, :] = np.interp(t, svi_times[j], svi_testlls[j])
tll_mean = np.mean(tllp, axis=0)
tll_std = np.std(tllp, axis=0)
lghdl, = plt.plot(t, tll_mean, c=snsm[4], lw=2)
lghdls.append(lghdl)
plt.plot(t, tll_mean+tll_std, c=snsm[4], ls='--', lw=2, alpha=0.4)
plt.plot(t, tll_mean-tll_std, c=snsm[4], ls='--', lw=2, alpha=0.4)
plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[4], alpha=0.3)
lglbls.append('SVI')


movb_times = []
movb_testlls = []
for j in range(len(mcmc_run_tags)):
    mtag = mcmc_run_tags[j]
    tr = np.genfromtxt(movbbasename+'-'+mtag+'-trace.log')
    movb_times.append(tr[:, 0])
    movb_testlls.append(tr[:, 1])
minTime = np.amin(np.array(map(np.amin, movb_times)))
maxTime = np.amax(np.array(map(np.amax, movb_times)))
t = np.logspace(np.log10(minTime), np.log10(maxTime), num=100)
tllp = np.zeros((len(movb_times), t.shape[0]))
for j in range(len(movb_times)):
    tllp[j, :] = np.interp(t, movb_times[j], movb_testlls[j])
tll_mean = np.mean(tllp, axis=0)
tll_std = np.std(tllp, axis=0)
lghdl, = plt.plot(t, tll_mean, c=snsm[5], lw=2)
lghdls.append(lghdl)
plt.plot(t, tll_mean+tll_std, c=snsm[5], ls='--', lw=2, alpha=0.4)
plt.plot(t, tll_mean-tll_std, c=snsm[5], ls='--', lw=2, alpha=0.4)
plt.fill_between(t, tll_mean-tll_std, tll_mean+tll_std, facecolor=snsm[5], alpha=0.3)
lglbls.append('moVB')

plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Test Log Likelihood')
plt.xlim([1e-3, 10])
plt.ylim([-11.2, -6])
plt.axes().set_yticklabels(map(int, plt.axes().get_yticks().tolist())) #these two commands for some reason help formatting the axis tick labels...)
plt.axes().set_xticklabels(map(float, plt.axes().get_xticks().tolist())) #these two commands for some reason help formatting the axis tick labels...)
plt.legend(lghdls, lglbls, loc=2)
plt.show()
