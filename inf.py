# -*- coding: utf-8 -*-

import numpy as np
from . import lik
from .import cov
from .tools import jitchol, solve_chol


#%%---------------------------------------------------------------------------#
class Inference(object):
    '''
    Base class for inference. Defines several tool methods in it.
    '''

#%%---------------------------------------------------------------------------#
class EP(Inference):
    '''
    Expectation Propagation approximation to the posterior Gaussian Process.
    '''
    
#%%---------------------------------------------------------------------------#
class Exact(Inference):
    '''
    Exact inference for a GP with Gaussian likelihood. Compute a parametrization
    of the posterior, the negative log marginal likelihood and its derivatives
    w.r.t. the hyperparameters.
    '''

    #%%-----------------------------------------------------------------------#
    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        
        assert isinstance(likfunc, lik.Gauss), \
               'Exact inference only possible with Gaussian likelihood'
        
        n, _ = x.shape
        
        assert n == y.size

        # evaluate covariance matrix
        K = covfunc.get_cov(x=x)  

        # evaluate mean vector          
        m = meanfunc.get_mean(x)   

        # noise variance of likGauss
        sn2 = np.exp(2 * likfunc.hyp[0])                       
        
        # Cholesky factor of covariance with noise
        L = jitchol((K / sn2) + np.eye(n)).T  
          
        alpha = solve_chol(L, (y - m)) / sn2     

        # sqrt of noise precision vector                                  
        sW = np.ones((n, 1)) / np.sqrt(sn2)
        
        post = (alpha, L, sW)
                    
        argouts = post
        
        # do we want the marginal likelihood?
        if nargout >= 2:
            
            # -log marg lik   
            nlZ, = (np.dot((y - m).T, alpha) / 2. 
                   + np.log(np.diag(L)).sum() 
                   + n * np.log(2 * np.pi * sn2) / 2.).ravel() 
            
            argouts = (nlZ, post)
            
            # do we want derivatives?
            if nargout > 2:     

                # precompute for convenience
                Q = solve_chol(L, np.eye(n)) / sn2 - np.dot(alpha, alpha.T)
                
                dlik = sn2 * np.trace(Q)
                
                dcov = np.fromiter(((Q * covfunc.get_cov_der(x=x, der=ii)).sum() 
                                      / 2. for ii in range(covfunc.hyp.size)), dtype=float)

                dmean = np.fromiter((np.dot(-meanfunc.get_mean_der(x, ii).T, 
                            alpha).ravel() for ii in range(meanfunc.hyp.size)), dtype=float).ravel()
                
                dnlZ = (dmean, dcov, dlik)

                argouts = (nlZ, dnlZ, post)
       

        return argouts
    
#%%---------------------------------------------------------------------------#
class Exact_FITC(Inference):
    '''
    FITC approximation to the posterior Gaussian process. The function is
    equivalent to infExact with the covariance function:
    Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku' * inv(Quu) * Ku;
    where Ku and Kuu are covariances w.r.t. to inducing inputs xu, snu2 = sn2/1e6
    is the noise of the inducing inputs and Quu = Kuu + snu2*eye(nu).
    '''
    #%%-----------------------------------------------------------------------#
    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        
        assert isinstance(likfunc, lik.Gauss), 'Exact inference only possible with Gaussian likelihood'
        assert isinstance(covfunc, cov.FITC), 'Only cov.FITC supported.'
        
        # evaluate covariance matrix
        diagK, Kuu, Ku = covfunc.get_cov(x=x)  
        
        # evaluate mean vector
        m = meanfunc.get_mean(x)
                                
        n, D = x.shape
        nu = Kuu.shape[0]
        
        # noise variance of likGauss
        sn2 = np.exp(2 * likfunc.hyp[0])   
        
        # hard coded inducing inputs noise            
        snu2 = 1.e-6 * sn2                                        
        
        # Kuu + snu2*I = Luu'*Luu
        Luu = jitchol(Kuu + snu2 * np.eye(nu)).T 

        # V = inv(Luu')*Ku => V'*V = Q                 
        V = np.linalg.solve(Luu.T, Ku)  
              
        # g + sn2 = diag(K) + sn2 - diag(Q)
        g_sn2 = diagK + sn2 - (V * V).sum(axis=0).reshape(-1, 1)
        
        # Lu'*Lu=I+V*diag(1/g_sn2)*V'
        Lu = jitchol(np.eye(nu) + np.dot((V / np.tile(g_sn2.T, (nu, 1))), V.T)).T
        
        r = (y - m) / np.sqrt(g_sn2)
        
        be = np.linalg.solve(Lu.T, np.dot(V, (r / np.sqrt(g_sn2))))
        
        # inv(Kuu + snu2*I) = iKuu
        iKuu  = solve_chol(Luu, np.eye(nu))                       

        # return the posterior parameters
        alpha = np.linalg.solve(Luu, np.linalg.solve(Lu, be)) 
        
        # Sigma-inv(Kuu)
        L = solve_chol(np.dot(Lu, Luu), np.eye(nu)) - iKuu   
        
        # unused for FITC prediction with gp.m
        sW = np.ones((n, 1)) / np.sqrt(sn2)    
        
        post = (alpha, L, sW)                

        argouts = post
        
        # do we want the marginal likelihood?
        if nargout >= 2:
                                    
            nlZ, = (np.log(np.diag(Lu)).sum() 
                   + (np.log(g_sn2).sum() + n * np.log(2 * np.pi) 
                   + np.dot(r.T, r) - np.dot(be.T, be)) / 2.).ravel()
            
            argouts = (nlZ, post)
            
            # do we want derivatives?
            if nargout > 2: 

                
                # al = (Kt+sn2*eye(n))\y
                al = r / np.sqrt(g_sn2) - np.dot(V.T, np.linalg.solve(Lu,be)) / g_sn2 
                
                B = np.dot(iKuu, Ku)
                
                w = np.dot(B, al)
                
                W = np.linalg.solve(Lu.T, V / np.tile(g_sn2.T, (nu, 1)))
                
                dcov = np.zeros_like(covfunc.hyp)
                
                for ii in range(dcov.size):
                    
                    # eval cov deriv
                    [ddiagKi, dKuui, dKui] = covfunc.get_cov_der(x=x, der=ii)
                    
                    R = 2. * dKui - np.dot(dKuui, B)
                    
                    # diag part of cov deriv
                    v = ddiagKi - (R * B).sum(axis=0).reshape(-1, 1)

                    dcov[ii], = ((np.dot(ddiagKi.T, 1. / g_sn2) 
                                + np.dot(w.T, (np.dot(dKuui, w) - 2. * np.dot(dKui, al))) 
                                - np.dot(al.T, (v * al)) 
                                - np.dot((W * W).sum(axis=0).reshape(1, -1), v) 
                                - (np.dot(R, W.T) * np.dot(B, W.T)).sum() )
                                / 2.).ravel()

                dlik = sn2 * (((1. / g_sn2)).sum() - (W * W).sum() - np.dot(al.T, al))
                
                dKuui = 2 * snu2
                
                R = -dKuui * B
                
                # diag part of cov deriv
                v = -(R * B).sum(axis=0).reshape(-1, 1)
                
                dlik += ((np.dot(w.T, np.dot(dKuui, w)) 
                        - np.dot(al.T, (v * al)) 
                        - np.dot((W*W).sum(axis=0).reshape(1, -1), v) 
                        - (np.dot(R ,W.T) * np.dot(B, W.T)).sum() ) 
                        / 2.)
                
                dlik = dlik.ravel()
                                
                dmean = np.fromiter((np.dot(-meanfunc.get_mean_der(x, ii).T, 
                            al).ravel() for ii in range(meanfunc.hyp.size)), dtype=float).ravel()
                
                dnlZ = (dmean, dcov, dlik)

                argouts = (nlZ, dnlZ, post)

        return argouts
#%%---------------------------------------------------------------------------#



