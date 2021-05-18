import torch
import numpy as np
import math
from math import pi
from scipy.special import logsumexp


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, device = 'cuda', mu_init=None, var_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.device = device

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False).double()
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False).double()

        if self.var_init is not None:
            assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.var = torch.nn.Parameter(self.var_init, requires_grad=False).double()
        else:
            self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False).double()

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1./self.n_components).double()
        
        self.to(self.device)

        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x


    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, sum_data=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if (self.log_likelihood.abs() == float("Inf")) or (self.log_likelihood == float("nan")):
                # When the log-likelihood assumes inane values, reinitialize model
                #print('self.log_likelihood')
                #print(self.log_likelihood)
                self.__init__(self.n_components,
                    self.n_features,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            print(p_k)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, sum_data=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu
        prec = torch.rsqrt(self.var)
        #print(prec)
        #print(self.mu)

        log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

        return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n,d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        x2 = (resp * x * x).sum(0, keepdim=True) / pi
        mu2 = mu * mu
        xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)
        #print(var.size())

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        #print(var.size())

        assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features), (1, self.n_features), (1, 1, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if var.size() == (self.n_components, self.n_features) or var.size() == (1, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features) or var.size() == (1, 1, self.n_features):
            self.var.data = var
            self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """

        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi

        
        
class GaussianTiedMixture(GaussianMixture):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init.double(), requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features).double(), requires_grad=False)

        if self.var_init is not None:
            assert self.var_init.size() == (1, 1, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.var = torch.nn.Parameter(self.var_init.double(), requires_grad=False)
        else:
            self.var = torch.nn.Parameter(torch.ones(1, 1, self.n_features).double() *0.3, requires_grad=False)

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1).double(), requires_grad=False).fill_(1./self.n_components)
        
        self.to(self.device)

        self.params_fitted = False


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
        
        dist = x - mu

        x2 = (resp * dist * dist).sum(1, keepdim=True)
        var = x2.sum(0, keepdim=True) / x.shape[0]
        #mu2 = mu * mu
        #xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        #var = x2 - 2 * xmu + mu2 + self.eps
        #var = var.mean(1,keepdim=True)

        pi = pi / x.shape[0]

        return pi, mu, var




    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        print(var.size())

        assert var.size() in [(1, self.n_features), (1, 1, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (1, self.n_features, self.n_components, self.n_features)

        if var.size() == (1, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, 1, self.n_features):
            self.var.data = var

        
       
        

class GaussianLaplaceMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, g_components, l_components, n_features, device = 'cuda', mu_init=None, var_init=None, a_init=None, b_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianLaplaceMixture, self).__init__()

        self.g_components = g_components
        self.l_components = l_components
        self.n_features = n_features
        self.device = device

        self.mu_init = mu_init
        self.var_init = var_init
        self.a_init = a_init
        self.b_init = b_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.g_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.g_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.g_components, self.n_features), requires_grad=False)

        if self.var_init is not None:
            assert self.var_init.size() == (1, self.g_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.g_components, self.n_features)
            # (1, k, d)
            self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
        else:
            self.var = torch.nn.Parameter(torch.ones(1, self.g_components, self.n_features), requires_grad=False)
        
        if self.a_init is not None:
            assert self.a_init.size() == (1, self.l_components, self.n_features), "Input a_init does not have required tensor dimensions (1, %i, %i)" % (self.l_components, self.n_features)
            # (1, k, d)
            self.a = torch.nn.Parameter(self.a_init, requires_grad=False)
        else:
            self.a = torch.nn.Parameter(torch.randn(1, self.l_components, self.n_features), requires_grad=False)

        if self.b_init is not None:
            assert self.b_init.size() == (1, self.l_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.l_components, self.n_features)
            # (1, k, d)
            self.b = torch.nn.Parameter(self.b_init, requires_grad=False)
        else:
            self.b = torch.nn.Parameter(torch.ones(1, self.l_components, self.n_features), requires_grad=False)

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.l_components + self.g_components, 1), requires_grad=False).fill_(1./(self.l_components + self.g_components))
        
        self.to(self.device)

        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x


    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * (self.l_components + self.g_components) + self.n_features + (self.l_components + self.g_components) - 1

        bic = -2. * self.__score(x, sum_data=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        #if not warm_start and self.params_fitted:
        #    self._init_params()

        x = self.check_size(x)

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var
            a_old = self.a
            b_old = self.b

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if (self.log_likelihood.abs() == float("Inf")) or (self.log_likelihood == float("nan")):
                # When the log-likelihood assumes inane values, reinitialize model
                self._init_params()

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)
                self.__update_a(a_old)
                self.__update_b(b_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        #print(weighted_log_prob)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, sum_data=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu
        prec = torch.rsqrt(self.var)
        #print(prec)
        #print(self.mu)

        log_gp = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_gdet = torch.sum(torch.log(prec), dim=2, keepdim=True)
        
        log_prob_g = -.5 * (self.n_features * np.log(2. * pi) + log_gp) + log_gdet
        
        a = self.a
        rb = 1.0 / self.b
        
        log_lp = torch.sum(torch.abs(x - a) * rb, dim=2, keepdim=True)
        log_ldet = torch.sum(torch.log(rb), dim=2, keepdim=True)
        
        log_prob_l = -(self.n_features * np.log(2) + log_lp) + log_ldet

        return torch.cat((log_prob_g, log_prob_l), dim = 1)


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n,d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu_a = torch.sum(resp * x, dim=0, keepdim=True) / pi
        mu = mu_a[:,:self.g_components,:]
        a = mu_a[:,self.g_components:,:]

        x2 = (resp[:,:self.g_components,:] * x * x).sum(0, keepdim=True) / pi[:,:self.g_components,:]
        mu2 = mu * mu
        xmu = (resp[:,:self.g_components,:] * mu * x).sum(0, keepdim=True) / pi[:,:self.g_components,:]
        var = x2 - 2 * xmu + mu2 + self.eps
        
        b = (resp[:,self.g_components:,:] * torch.abs(x-a)).sum(0, keepdim=True) / pi[:,self.g_components:,:]

        pi = pi / x.shape[0]

        return pi, mu, var, a, b


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var, a, b = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)
        self.__update_a(a)
        self.__update_b(b)


    def __score(self, x, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert mu.size() in [(self.g_components, self.n_features), (1, self.g_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.g_components, self.n_features, self.g_components, self.n_features)

        if mu.size() == (self.g_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.g_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """

        assert var.size() in [(self.g_components, self.n_features), (1, self.g_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.g_components, self.n_features, self.g_components, self.n_features)

        if var.size() == (self.g_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.g_components, self.n_features):
            self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """

        assert pi.size() in [(1, (self.l_components + self.g_components), 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, (self.l_components + self.g_components), 1)

        self.pi.data = pi
        
    def __update_a(self, a):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert a.size() in [(self.l_components, self.n_features), (1, self.l_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.l_components, self.n_features, self.l_components, self.n_features)

        if a.size() == (self.l_components, self.n_features):
            self.a = a.unsqueeze(0)
        elif a.size() == (1, self.l_components, self.n_features):
            self.a.data = a


    def __update_b(self, b):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """

        assert b.size() in [(self.l_components, self.n_features), (1, self.l_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.l_components, self.n_features, self.l_components, self.n_features)

        if b.size() == (self.l_components, self.n_features):
            self.b = b.unsqueeze(0)
        elif b.size() == (1, self.l_components, self.n_features):
            self.b.data = b

            
            
        
class GaussianLaplaceTiedMixture(GaussianTiedMixture):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, g_components, l_components, n_features, device = 'cuda', mu_init=None, var_init=None, a_init=None, b_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianLaplaceTiedMixture, self).__init__(g_components + l_components, n_features, device = device, mu_init=mu_init, var_init=var_init, eps=1.e-6)

        self.g_components = g_components
        self.l_components = l_components

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu[:,:self.g_components,:]
        prec = torch.rsqrt(self.var)
        #print(prec)
        #print(self.mu)

        log_gp = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_gdet = torch.sum(torch.log(prec), dim=2, keepdim=True)
        
        log_prob_g = -.5 * (self.n_features * np.log(2. * pi) + log_gp) + log_gdet
        
        a = self.mu[:,self.g_components:,:]
        rb = torch.rsqrt(self.var*0.99)
        log_lp = torch.sum((a * a + x * x - 2 * x * a) * (rb ** 2), dim=2, keepdim=True)
        #log_lp = torch.sum(torch.abs(x - a) * rb, dim=2, keepdim=True)
        log_ldet = torch.sum(torch.log(rb), dim=2, keepdim=True)
        
        #log_prob_l = -(self.n_features * np.log(2) + log_lp) + log_ldet
        log_prob_l = -.5 * (self.n_features * np.log(2. * pi) + log_lp) + log_ldet

        return torch.cat((log_prob_g, log_prob_l), dim = 1)


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)
        resp_g = resp[:,:self.g_components,:]
        resp_l = resp[:,self.g_components:,:]

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu_g = torch.sum(resp_g * x, dim=0, keepdim=True) / pi[:,:self.g_components,:]
        
        mu_l_frac = resp_l / ((x - self.mu[:,self.g_components:,:]).abs() + self.eps)
        #print(mu_l_frac.size())
        mu_l = torch.sum(mu_l_frac * x, dim=0, keepdim=True) 
        #print(mu_l.size())
        mu_l = mu_l / (torch.sum(mu_l_frac, dim=0, keepdim=True) + self.eps)
        #print(mu_l.size())
        #print(mu_g.size())
        mu_l = torch.sum(resp_l * x, dim=0, keepdim=True) / pi[:,self.g_components:,:]
        mu = torch.cat((mu_g, mu_l), dim = 1)
        #print(mu.size())
        
        dist = x - mu
        
        dist_g = dist[:,:self.g_components,:]
        
        
        dist_l = dist[:,self.g_components:,:]
        

        x2 = (resp_g * dist_g * dist_g).sum(1, keepdim=True)
        var = x2.sum(0, keepdim=True) 
        
        x1 = (resp_l * dist_l * dist_l).sum(1, keepdim=True)
        var_l = x1.sum(0, keepdim=True) 
        
        
        var = var + var_l * (0.95)
        var = var / x.shape[0]
        #var_sqrt = var_sqrt / (2 * x.shape[0])
        #var = 2 * var_sqrt * var_sqrt
        #mu2 = mu * mu
        #xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        #var = x2 - 2 * xmu + mu2 + self.eps
        #var = var.mean(1,keepdim=True)

        pi = pi / x.shape[0]

        return pi, mu, var


class GaussianLaplaceTiedAutoMixture(GaussianTiedMixture):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, g_components, l_components, n_features, device = 'cuda', mu_init=None, var_init=None, alpha_init = None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        
        super(GaussianLaplaceTiedAutoMixture, self).__init__(g_components + l_components, n_features, device = device, mu_init=mu_init, var_init=var_init, eps=1.e-6)
        if alpha_init is None:
            self.alpha_init = 1.
        else:
            self.alpha_init = alpha_init
        self.alpha = torch.nn.Parameter(torch.tensor(self.alpha_init).double().to(device), requires_grad=False)

        self.g_components = g_components
        self.l_components = l_components

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu[:,:self.g_components,:]
        prec = torch.rsqrt(self.var)
        #print(prec)
        #print(self.mu)

        log_gp = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_gdet = torch.sum(torch.log(prec), dim=2, keepdim=True)
        
        log_prob_g = -.5 * (self.n_features * np.log(2. * pi) + log_gp) + log_gdet
        
        a = self.mu[:,self.g_components:,:]
        rb = torch.rsqrt(self.var*self.alpha)
        log_lp = torch.sum((a * a + x * x - 2 * x * a) * (rb ** 2), dim=2, keepdim=True)
        #log_lp = torch.sum(torch.abs(x - a) * rb, dim=2, keepdim=True)
        log_ldet = torch.sum(torch.log(rb), dim=2, keepdim=True)
        
        #log_prob_l = -(self.n_features * np.log(2) + log_lp) + log_ldet
        log_prob_l = -.5 * (self.n_features * np.log(2. * pi) + log_lp) + log_ldet

        return torch.cat((log_prob_g, log_prob_l), dim = 1)


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)
        resp_g = resp[:,:self.g_components,:]
        resp_l = resp[:,self.g_components:,:]

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu_g = torch.sum(resp_g * x, dim=0, keepdim=True) / pi[:,:self.g_components,:]
        
        
        mu_l = torch.sum(resp_l * x, dim=0, keepdim=True) / pi[:,self.g_components:,:]
        mu = torch.cat((mu_g, mu_l), dim = 1)
        #print(mu.size())
        
        dist = x - mu
        
        dist_g = dist[:,:self.g_components,:]
        
        
        dist_l = dist[:,self.g_components:,:]
        

        x2 = (resp_g * dist_g * dist_g).sum(1, keepdim=True)
        var = x2.sum(0, keepdim=True) 
        
        x1 = (resp_l * dist_l * dist_l).sum(1, keepdim=True)
        var_l = x1.sum(0, keepdim=True) 
        
        
        var = var + var_l / self.alpha
        var = var / x.shape[0]
        alpha = (var_l / self.var).mean() / (pi[:,self.g_components:,:].sum())
        print(alpha)
        self.alpha.data = alpha
        #var_sqrt = var_sqrt / (2 * x.shape[0])
        #var = 2 * var_sqrt * var_sqrt
        #mu2 = mu * mu
        #xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        #var = x2 - 2 * xmu + mu2 + self.eps
        #var = var.mean(1,keepdim=True)

        pi = pi / x.shape[0]

        return pi, mu, var