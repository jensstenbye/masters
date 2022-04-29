import torch
from torch.distributions import Distribution, Gamma, Poisson


#Negative binomial distribution class
'''Læs op på parametriseringen igen, skal den med i teori når den er fra scVI?'''
class NegativeBinomial(Distribution):
    def __init__(self, mu: torch.Tensor, theta: torch.Tensor):
        super().__init__()  
        self.eps = 1e-8 
        self.mu = mu                                #Mean of distribution (Tensor shape batch x genes)
        self.theta = theta.expand_as(mu)    #Dispersion of distribution(shape batch x genes)
        self.var = self.mu + (self.mu ** 2) / self.theta 
        
    
    #Returns loglikelihood of data given mean and dispersion for NB distribution
    def _log_nb(self, data: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
        log_L = (theta * (torch.log(theta + eps) - torch.log(theta + mu + eps)) 
                 + data * (torch.log(mu + eps) - torch.log(theta + mu + eps)) 
                 + torch.lgamma(data + theta +eps)
                 - torch.lgamma(theta + eps)
                 - torch.lgamma(data + 1)
                )

        return torch.sum(log_L)
       
    #Return negative log likelihood of data using mean and shape parameters
    def logL(self, data):
        return self._log_nb(data = data, mu = self.mu, theta = self.theta, eps = self.eps)

    ####Below not important for model running###
    def _gamma(self):
        #Returns gamme distribution
        return torch.distributions.Gamma(concentration=self.theta, rate=self.mu)
     
    #Sample, used for verification 
    def sample(self, sample_shape):
        #Use poisson-gamma mixture  
        with torch.no_grad():
            #Get gamma distribution for sample
            gamma_d = self._gamma()
            #Extract poisson rates
            p_rates = gamma_d.sample(sample_shape)
            
            #Set max value of parameters to prevent bugs
            clamp_p_rates = torch.clamp(p_rates, max = 1e8)
            
            #Extract counts by sampling poisson using the rates
            counts = torch.distributions.Poisson(clamp_p_rates).sample()
            return counts


'''Kig på denne her'''
class NegativeBinomialLoss(torch.nn.Module):
    def __init__(self):
        super(NegativeBinomialLoss, self).__init__()
        self.distribution = NegativeBinomial

    def forward(self, mean, dispersion, size_factors, target_data):
        #Create a NB distribution with fitted mean/dispersion
        #And get likelihood of target_data given this distribution
        fit_distribution = self.distribution(size_factors*mean, dispersion)
        neg_logL = -fit_distribution.logL(target_data) 
        return neg_logL