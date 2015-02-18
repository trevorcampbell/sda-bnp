#ifndef __HDP_IMPL_HPP

template<class Model>
VarHDP<Model>::VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, double eta, uint32_t T, uint32_t K) : model(model), test_data(test_data), gam(gam), alpha(alpha), eta(eta), T(T), K(K){
	this->M = model.getStatDimension();
	this->N = train_data.size();
	this->Nt = test_data.size();

	for (uint32_t i = 0; i < N; i++){
		this->Nl.push_back(train_data[i].size());
		train_stats.push_back(MXd::Zero(Nl.back(), M));
		for (uint32_t j = 0; j < Nl.back(); j++){
			train_stats.back().row(j) = this->model.getStat(train_data[i][j]).transpose();
		}
	}
	for (uint32_t i = 0; i < Nt; i++){
		this->Ntl.push_back(test_data[i].size());
	}

	//seed random gen
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	nu = VXd::Zero(T);
	eta = MXd::Zero(T, M);
	u = v = VXd::Zeros(T-1); 
	for (uint32_t i = 0; i < N; i++){
		a.push_back(VXd::Zeros(K-1));
		b.push_back(VXd::Zeros(K-1));
		zeta.push_back(MXd::Zero(Nl[i], K));
		phi.push_back(MXd::Zero(K, T));
	}

}

template<class Model>
void VarHDP<Model>::init(){
	//initialize topic word weights
	std::gamma_distribution<> gamdist; //using alpha=1.0, eta =1.0
	for(uint32_t i = 0; i < T; i++){
		for(uint32_t j =0; j < M; j++){
			eta(i, j) = gamdist(rng)*D*100.0/(T*M);
		}
	}

	u = VXd::Ones(T-1);
	v = gam*VXd::Ones(T-1);

	for(uint32_t i =0; i < N; i++){
	}

}


template<class Model>
void VarHDP<Model>::run(bool computeTestLL = false, double tol = 1e-6){
	//clear any previously stored results
	times.clear();
	objs.clear();
	testlls.clear();

	//create objective tracking vars
	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();

	//start the timer
	Timer cpuTime, wallTime;
	cpuTime.start();
	wallTime.start();

	//initialize the variables
	init();

	//loop on variational updates
	while(diff > tol){
		//update the local distributions
		updateLocalDists(tol);
		//update the global distribution
		updateGlobalDist();

		prevobj = obj;
		//store the current time
		times.push_back(cpuTime.get());
		//compute the objective
		for (uint32_t i =0; i < N; i++){
			obj += computeLocalObjective(idx);
		}
		obj += computeGlobalObjective();
		//save the objective
		objs.push_back(obj);
		//compute the obj diff
		diff = fabs((obj - prevobj)/obj);
		//if test likelihoods were requested, compute those (but pause the timer first)
		if (computeTestLL){
			cpuTime.stop();
			double testll = computeTestLogLikelihood();
			testlls.push_back(testll);
			cpuTime.start();
			std::cout << "obj: " << obj << " testll: " << testll << std::endl;
		} else {
			std::cout << "obj: " << obj << std::endl;
		}
	}
	//done!
	return;

}

template<class Model>
void VarHDP<Model>::getResults(){
}

template<class Model>
void VarHDP<Model>::updateLocalDists(double tol){
	//zero out the sufficient stats
//TODO		

	//run variational updates on the local params
	for (uint32_t i = 0; i < N; i++){
		//create objective tracking vars
		double diff = 10.0*tol + 1.0;
		double obj = std::numeric_limits<double>::infinity();
		double prevobj = std::numeric_limits<double>::infinity();

		while(diff > tol){
			updateLocalWeightDist(idx);
			updateLocalLabelDist(idx);
			updateLocalCorrespondenceDist(idx);
			prevobj = obj;
			obj = computeLocalObjective(idx);
			//compute the obj diff
			diff = fabs((obj - prevobj)/obj);
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalWeightDist(uint32_t idx){
	//Update a, b, and psisum
	psiabsum[idx] = VXd::Zero(K);
	double psibk = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
		a[idx](k) = 1.0+zetasum(k);
		b[idx](k) = alpha;
		for (uint32_t j = k+1; j < K; j++){
			b[idx](k) += zetasum(j);
		}
    	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    	psiabsum[idx](k) = psiak + psibk;
    	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}
	psiabsum[idx](K-1) = psibk;
}

template<class Model>
void VarHDP<Model>::updateLocalLabelDist(uint32_t idx){
	//update the label distribution
	zetasum[idx] = VXd::Zero(K);
	zetaTsum[idx] = MXd::Zero(K, M);
	for (uint32_t i = 0; i < Nl[idx]; i++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) = psiabsum[idx](k) - phiNsum[idx](k);
			for (uint32_t j = 0; j < M; j++){
				zeta[idx](i, k) -= train_stats[idx](i, j)*phiEsum[idx](k, j);
			}
			logpmax = (zeta[idx](i, k) > logpmax ? zeta[idx](i, k) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) -= logpmax;
			zeta[idx](i, k) = exp(zeta[idx](i, k));
			psum += zeta[idx](i, k);
		}
		//normalize
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) /= psum;
		}
		//update the zetasum stats
		zetasum[idx] += zeta[idx].row(i).transpose();
		for(uint32_t k = 0; k < K; k++){
			zetaTsum[idx].row(k) += zeta[idx](i, k)*train_stats[idx].row(i);
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalCorrespondenceDist(uint32_t idx){
	//update the correspondence distribution
	phiNsum[idx] = VXd::Zero(K);
	phiEsum[idx] = MXd::Zero(K, M);
	for (uint32_t k = 0; k < K; k++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) = psiuvsum[idx](t) - zetasum[idx](k)*dlogh_dnu(t);
			for (uint32_t j = 0; j < M; j++){
				phi[idx](k, t) -= zetaTsum[idx](k, j)*dlogh_deta(t, j);
			}
			logpmax = (phi[idx](k, t) > logpmax ? phi[idx](k, t) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) -= logpmax;
			phi[idx](k, t) = exp(phi[idx](k, t));
			psum += phi[idx](k, t);
		}
		//normalize
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) /= psum;
		}
		//update the phisum stats
		for(uint32_t t = 0; t < T; t++){
			phiNsum[idx](k) += phi[idx](k, t)*dlogh_dnu(t);
			phiEsum[idx].row(k) += phi[idx](k, t)*dlogh_deta.row(t);
		}
	}
}



template<class Model>
void VarHDP<Model>::updateGlobalDist(){
	updateGlobalWeightDist();
	updateGlobalParamDist();
}

template<class Model>
void VarHDP<Model>::updateGlobalWeightDist(){
	//Update u, v, and psisum
	psiuvsum[idx] = VXd::Zero(T);
	double psivk = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
		u[idx](t) = 1.0+phisum(t);
		v[idx](t) = gam;
		for (uint32_t j = t+1; j < T; j++){
			v[idx](t) += phisum(j);
		}
    	double psiuk = digamma(u(k)) - digamma(u(k)+v(k));
    	psiabsum[idx](k) = psiuk + psivk;
    	psivk += digamma(v(k)) - digamma(u(k)+v(k));
	}
	psiuvsum[idx](T-1) = psivk;
}

template<class Model>
void VarHDP<Model>::updateGlobalParamDist(){
	for (uint32_t t = 0; t < T; t++){
		nu(t) = model.getNu0() + phizetasum(t);
		for (uint32_t j = 0; j < M; j++){
			eta(t, j) = model.getEta0()(j) + phizetaTsum.row(t, j);
		}
	}
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);
}

template<class Model>
double VarHDP<Model>::computeGlobalObjective(){

}

template<class Model>
double VarHDP<Model>::computeLocalObjective(uint32_t idx){

}

template<class Model>
double VarHDP<Model>::computeTestLogLikelihood(){

}


def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])

def expect_log_sticks(sticks):
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks



class hdp_hyperparameter:
    def __init__(self, alpha_a, alpha_b, gamma_a, gamma_b, hyper_opt=False):
        self.m_alpha_a = alpha_a
        self.m_alpha_b = alpha_b
        self.m_gamma_a = gamma_a
        self.m_gamma_b = gamma_b
        self.m_hyper_opt = hyper_opt

class suff_stats:
    def __init__(self, T, size_vocab):
        self.m_var_sticks_ss = np.zeros(T)
        self.m_var_logp0_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros((T, size_vocab))

    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_logp0_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)



class hdp:
    ''' hdp model using john's new stick breaking'''
    def __init__(self, T, K, D,  size_vocab, eta, hdp_hyperparam):
        ''' this follows the convention of the HDP paper'''
        ''' gamma, first level concentration '''
        ''' alpha, second level concentration '''
        ''' eta, the topic Dirichlet '''
        ''' T, top level truncation level '''
        ''' K, second level truncation level '''
        ''' size_vocab, size of vocab'''
        ''' hdp_hyperparam, the hyperparameter of hdp '''

        self.m_hdp_hyperparam = hdp_hyperparam

        self.m_T = T
        self.m_K = K # for now, we assume all the same for the second level truncation
        self.m_size_vocab = size_vocab

        self.m_beta = np.random.gamma(1.0, 1.0, (T, size_vocab)) * D*100/(T*size_vocab)
        self.m_eta = eta

        self.m_alpha = hdp_hyperparam.m_alpha_a/hdp_hyperparam.m_alpha_b
        self.m_gamma = hdp_hyperparam.m_gamma_a/hdp_hyperparam.m_gamma_b
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = self.m_gamma

        # variational posterior parameters for hdp
        self.m_var_gamma_a = hdp_hyperparam.m_gamma_a
        self.m_var_gamma_b = hdp_hyperparam.m_gamma_b

def em_on_large_data(self, filename, var_converge, fresh):
        ss = suff_stats(self.m_T, self.m_size_vocab)
        ss.set_zero()

        # prepare all needs for a single doc
        Elogbeta = dirichlet_expectation(self.m_beta) # the topics
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        likelihood = 0.0
        for line in file(filename):
            doc = parse_line(line)
            likelihood += self.doc_e_step(doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=fresh)

        # collect the likelihood from other parts
        # the prior for gamma
        if self.m_hdp_hyperparam.m_hyper_opt:
            log_gamma = sp.psi(self.m_var_gamma_a) -  np.log(self.m_var_gamma_b)
            likelihood += self.m_hdp_hyperparam.m_gamma_a * log(self.m_hdp_hyperparam.m_gamma_b) \
                    - sp.gammaln(self.m_hdp_hyperparam.m_gamma_a)

            likelihood -= self.m_var_gamma_a * log(self.m_var_gamma_b) \
                    - sp.gammaln(self.m_var_gamma_a)

            likelihood += (self.m_hdp_hyperparam.m_gamma_a - self.m_var_gamma_a) * log_gamma \
                    - (self.m_hdp_hyperparam.m_gamma_b - self.m_var_gamma_b) * self.m_gamma
        else:
            log_gamma = np.log(self.m_gamma)

       # the W/sticks part
        likelihood += (self.m_T-1) * log_gamma
        dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
        likelihood += np.sum((np.array([1.0, self.m_gamma])[:,np.newaxis] - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(self.m_var_sticks, 0))) - np.sum(sp.gammaln(self.m_var_sticks))

        # the beta part
        likelihood += np.sum((self.m_eta - self.m_beta) * Elogbeta)
        likelihood += np.sum(sp.gammaln(self.m_beta) - sp.gammaln(self.m_eta))
        likelihood += np.sum(sp.gammaln(self.m_eta*self.m_size_vocab) - sp.gammaln(np.sum(self.m_beta, 1)))

        self.do_m_step(ss) # run m step
        return likelihood


def doc_e_step(self, doc, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=False):

        Elogbeta_doc = Elogbeta[:, doc.words]
        v = np.zeros((2, self.m_K-1))

        phi = np.ones((doc.length, self.m_K)) * 1.0/self.m_K

        # the following line is of no use
        Elogsticks_2nd = expect_log_sticks(v)

        likelihood = 0.0
        old_likelihood = -1e1000
        converge = 1.0
        eps = 1e-100

        iter = 0
        max_iter = 100
        #(TODO): support second level optimization in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi
            if iter < 3 and fresh:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)

           # phi
            if iter < 3:
                phi = np.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc.counts))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < 0:
                print "warning, likelihood is decreasing!"

            iter += 1

        # update the suff_stat ss
        ss.m_var_sticks_ss += np.sum(var_phi, 0)
        ss.m_var_logp0_ss += np.sum(np.log( (1.0-var_phi)+1.0e-200), 0)
        ss.m_var_beta_ss[:, doc.words] += np.dot(var_phi.T, phi.T * doc.counts)

        return(likelihood)

    def optimal_ordering(self, ss):
        s = [(a, b) for (a,b) in izip(ss.m_var_sticks_ss, range(self.m_T))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        ss.m_var_sticks_ss[:] = ss.m_var_sticks_ss[idx]
        ss.m_var_logp0_ss[:] = ss.m_var_logp0_ss[idx]
        ss.m_var_beta_ss[:] = ss.m_var_beta_ss[idx,:]

    def do_m_step(self, ss):
        self.optimal_ordering(ss)
        ## update top level sticks
        self.m_var_sticks[0] = ss.m_var_sticks_ss[:self.m_T-1] + 1.0
        var_phi_sum = np.flipud(ss.m_var_sticks_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

        ## save enk suff stats
        self.m_enk = np.array(ss.m_var_sticks_ss)
        self.m_lp0 = np.array(ss.m_var_logp0_ss)

        ## update topic parameters
        self.m_beta = self.m_eta + ss.m_var_beta_ss

        if self.m_hdp_hyperparam.m_hyper_opt:
            self.m_var_gamma_a = self.m_hdp_hyperparam.m_gamma_a + self.m_T - 1
            dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
            Elog1_W = sp.psi(self.m_var_sticks[1]) - dig_sum
            self.m_var_gamma_b = self.m_hdp_hyperparam.m_gamma_b - np.sum(Elog1_W)
            self.m_gamma = hdp_hyperparam.m_gamma_a/hdp_hyperparam.m_gamma_b

def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length/2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < doc.length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < doc.length]

   # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)

def hdp_to_lda(self):
        # compute the lda almost equivalent hdp.
        # alpha
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma

        # beta
        beta_sum = np.sum(self.m_beta, axis=1)
        beta = self.m_beta / beta_sum[:, np.newaxis]

        return (alpha, beta)




#define __HDP_IMPL_HPP
#endif /* __HDP_HPP */
