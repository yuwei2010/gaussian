# -*- coding: utf-8 -*-

import numpy as np

from . import inf
#%%---------------------------------------------------------------------------#
class Likelihood(object):
    
    """Base function for Likelihood function"""
    
#%%---------------------------------------------------------------------------#
class Gauss(Likelihood):
    '''
    Gaussian likelihood function for regression.

    :math:`Gauss(t)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(t-y)^2}{2\\sigma^2}}`,
    where :math:`y` is the mean and :math:`\\sigma` is the standard deviation.

    hyp = [ log_sigma ]
    '''
    
    def __init__(self, log_sigma=np.log(0.1)):
        
        self.hyp = np.array([log_sigma], dtype=float)
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):

        return self._hyp
    
    @hyp.setter
    def hyp(self, value):

        self._hyp = np.asarray(value, dtype=float)
        self.sn2 = np.exp(2. * self.hyp[0])
        
    #%%-----------------------------------------------------------------------#
    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        
        if inffunc is None:
            
            argouts = self.get_prediction(y=y, mu=mu, s2=s2, nargout=nargout)

        else:
            
            if isinstance(inffunc, inf.EP):    
                
                argouts = self.get_inference_ep(
                                    y=y, mu=mu, s2=s2, der=der, nargout=nargout)
                
            elif isinstance(inffunc, inf.Laplace):
                
                argouts = self.get_inference_laplace(
                                    y=y, mu=mu, s2=s2, der=der, nargout=nargout)
                
        return argouts
    
    #%%-----------------------------------------------------------------------#
    def get_prediction(self, y, mu, s2, nargout):
        
        sn2 = self.sn2
        
        if y is None:
            y = np.zeros_like(mu)
                   
        if s2 is not None and np.linalg.norm(s2) > 0:
            s2zero = False
        else:
            s2zero = True
        
        # log probability
        if s2zero:                   
            
            lp = -(y - mu)**2 / sn2 / 2 - np.log(2.* np.pi * sn2) / 2.
            s2 = np.zeros_like(s2)
            
        else: 
            lp = self.evaluate(y=y, mu=mu, s2=s2, inffunc=inf.EP())
        
        if nargout == 1:
            return lp
        else:
            return (lp, mu, s2 + sn2)[:nargout]
    
    #%%-----------------------------------------------------------------------#
    def get_inference_ep(self, y, mu, s2, der, nargout):
        
        sn2 = self.sn2
        
        # no derivative mode
        if der is None:                           
            
            # log part function
            lZ = (-(y - mu)**2 / (sn2 + s2) / 2. 
                  - np.log(2 * np.pi * (sn2 + s2)) / 2.) 
                                 
            # 1st derivative w.r.t. mean
            dlZ  = (y - mu) / (sn2 + s2)                   
                    
            # 2nd derivative w.r.t. mean
            d2lZ = -1 / (sn2 + s2) 
            
            if nargout == 1:
                return lZ
            else:
                return (lZ, dlZ, d2lZ)[:nargout]
                       
        # derivative mode
        else:
            
            # deriv. w.r.t. hyp.lik                                           
            dlZhyp = (((y - mu)**2 / (sn2 + s2)) - 1) / (1 + (s2 / sn2)) 
             
            return dlZhyp     
        
    #%%-----------------------------------------------------------------------#
    def get_inference_laplace(self, y, mu, s2, der, nargout):
    
        sn2 = self.sn2
        
        # no derivative mode
        if der is None:                                  
        
            if y is None:
                y = 0
                
            ymmu = y - mu
            lp = -ymmu**2 / (2 * sn2) - (np.log(2 * np.pi * sn2) / 2.)
                            
            # dlp, derivative of log likelihood
            dlp = ymmu / sn2                           
                                          
            # d2lp, 2nd derivative of log likelihood
            d2lp = -np.ones_like(ymmu) / sn2
                    
            # d3lp, 3rd derivative of log likelihood
            d3lp = np.zeros_like(ymmu)
            
            return (lp, dlp, d2lp, d3lp)[:nargout]
                            
        # derivative mode
        else:
            
            # derivative of log likelihood w.r.t. hypers                                           
            lp_dhyp   = ((y - mu)**2 / sn2) - 1  
            
            # first derivative,
            dlp_dhyp  = 2 * (mu - y) / sn2
            
             # and also of the second mu derivative                     
            d2lp_dhyp = 2 * np.ones_like(mu) / sn2 
            
            return lp_dhyp, dlp_dhyp, d2lp_dhyp
        
#%%---------------------------------------------------------------------------#
class Erf(Likelihood):
    '''
    Error function or cumulative Gaussian likelihood function for binary
    classification or probit regression.

    :math:`Erf(t)=\\frac{1}{2}(1+erf(\\frac{t}{\\sqrt{2}}))=normcdf(t)`
    '''
#%%---------------------------------------------------------------------------#