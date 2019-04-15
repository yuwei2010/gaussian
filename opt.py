# -*- coding: utf-8 -*-
import numpy as np
import logging

from .tools import minimize

#%%---------------------------------------------------------------------------#
class Optimizer(object):
    
    def __init__(self, model):
        
        self.model = model
        
        self.logger = logging.getLogger(__name__)
        
    #%%-----------------------------------------------------------------------#  
    @property
    def slice_mean(self):
        
        return slice(0, self.model.mean.hyp.size)
    
    #%%-----------------------------------------------------------------------# 
    @property
    def slice_cov(self):
        
        return slice(self.model.mean.hyp.size, 
                    (self.model.mean.hyp.size + self.model.cov.hyp.size))
        
    #%%-----------------------------------------------------------------------# 
    @property
    def slice_lik(self):
        
        return slice((self.model.mean.hyp.size + self.model.cov.hyp.size), None)
    
    #%%-----------------------------------------------------------------------#
    def get_hypers(self):
        
        '''Convert all hyparameters in the model to an array'''
                
        return np.hstack([self.model.mean.hyp, 
                          self.model.cov.hyp, 
                          self.model.lik.hyp])  
    
    #%%-----------------------------------------------------------------------#    
    def apply_hypers(self, hyp_array):
        
        '''Apply the values in the input array to hyparameters of model.'''
                
        self.model.mean.hyp = hyp_array[self.slice_mean]
        self.model.cov.hyp = hyp_array[self.slice_cov]
        self.model.lik.hyp = hyp_array[self.slice_lik]

        return self
    
    #%%-----------------------------------------------------------------------#
    def optimize_nlZ(self, hyp_array):
        '''Find negative-log-marginal-likelihood'''
        
        self.apply_hypers(hyp_array)                
        nlZ, post = self.model.get_posterior(2)   
        
        return nlZ 
    
    #%%-----------------------------------------------------------------------#
    def optimize_nlZ_and_dnlZ(self, hyp_array):
        
        '''Find negative-log-marginal-likelihood and derivatives in one pass(faster)'''

        self.apply_hypers(hyp_array)                
        nlZ, dnlZ, post = self.model.get_posterior(3)

        return nlZ, np.hstack(dnlZ)
    
    #%%-----------------------------------------------------------------------#  
    def apply_config(self, **config):
        
        def get():
            for attr in ('mean', 'cov', 'lik'):
                
                curr_value = getattr(self.model, attr).hyp
                
                if config.get(attr, None):
                    
                    new_value = np.fromiter((np.random.uniform(*config[attr][i]) for
                                      i in range(curr_value.size)), dtype=float)
                    
                    self.logger.warning('Change %s.hyp to %s', attr, new_value)
                
                    yield new_value
                
                else:
                    yield curr_value
                    
        self.model.hypers = np.hstack(get())
                       
    #%%-----------------------------------------------------------------------#  
    def find_min(self, num_iters=200, num_restart=0, config=None, opt_depth=5):
        
        if num_restart > 0 and config is None:            
            config = dict(lik=[(-10, 10)])
        
        last_value = np.inf
        
        trails_counter = 0
        
        error_counter = 0
        
        depth_counter = 0
        
        if config:            
            np.random.seed()
        
        while 1:
            
            hyp_array = self.get_hypers()
    
            try:
                
                trails_counter += 1
                
                optimal_hyp, func_value, iterations = \
                    self.execute_optimize(hyp_array=hyp_array, num_iters=num_iters)
                    
                self.logger.warning("Number of iteration %g", iterations) 

                if trails_counter >= num_restart or depth_counter >= opt_depth:
                    break
                
                if last_value <= func_value:                    
                    self.model.hypers = hyp_array
                    break
                
                last_value = func_value
                depth_counter += 1
                
            except np.linalg.linalg.LinAlgError:
                
                error_counter += 1
                
                depth_counter = 0
                
                if not config and trails_counter > num_restart:
                    raise
                    
                elif config:
                    self.apply_config(**config)
                
        self.logger.warning("%d out of %d trails failed during optimization.", 
                                    error_counter, trails_counter) 
                
        return optimal_hyp, func_value, iterations  
    
#%%---------------------------------------------------------------------------#   
class Minimize(Optimizer):
    '''minimize by Carl Rasmussen (python implementation of "minimize" in GPML)'''
         
    def execute_optimize(self, hyp_array, num_iters):
        
        method = minimize
#        method = __import__('tools').minimize
        
        kwargs = dict(f=self.optimize_nlZ_and_dnlZ, X=hyp_array, length=num_iters)
        
        opt = method(**kwargs)

        optimal_hyp = opt[0]
        
        func_value = opt[1][-1]
        
        iterations = opt[2]
        
        return optimal_hyp, func_value, iterations

#%%---------------------------------------------------------------------------#   
class SCG(Optimizer):
    '''Scaled conjugent gradient (faster than CG)'''
         
    def execute_optimize(self, hyp_array, num_iters):
        
        method = __import__('tools').SCG
        
        kwargs = dict(f=self.optimize_nlZ_and_dnlZ, x=hyp_array, niters=num_iters)
        
        opt = method(**kwargs)

        optimal_hyp = opt[0]
        
        func_value = opt[1][-1]
        
        iterations = opt[2]
        
        return optimal_hyp, func_value, iterations

#%%---------------------------------------------------------------------------#  
class Scipy_Method(Optimizer):
    
    def execute_optimize(self, hyp_array, num_iters):

        method = __import__('scipy').optimize.minimize
                    
        fun = self.optimize_nlZ_and_dnlZ if self.__class__.jac == True else self.optimize_nlZ
                    
        kwargs = dict(fun=fun, x0=hyp_array, jac=self.__class__.jac,
                      method=self.__class__.method_name, 
                      options=dict(maxiter=num_iters, disp=False))
        
        opt = method(**kwargs)
        
        optimal_hyp = opt['x']
        
        func_value = opt['fun']

        iterations = opt.get('nit', opt['nfev'])
        
        return optimal_hyp, func_value, iterations    
                   
#%%---------------------------------------------------------------------------# 
class Simplex(Scipy_Method):
    '''Downhill simplex algorithm by Nelder-Mead'''
    
    method_name = 'Nelder-Mead'
    jac = False
    
#%%---------------------------------------------------------------------------#     
class Powell(Scipy_Method):
    '''modified Powell algorithm.'''
    method_name = 'Powell'    
    jac = False
    
#%%---------------------------------------------------------------------------#    
class COBYLA(Scipy_Method):
    '''Linear Approximation (COBYLA) algorithm'''
    method_name = 'COBYLA'
    jac = False
    
#%%---------------------------------------------------------------------------#     
class CG(Scipy_Method):
    '''Conjugent gradient'''
    method_name = 'CG'
    jac = True
    
#%%---------------------------------------------------------------------------#    
class Newton_CG(Scipy_Method):
    ''' Newton-CG algorithm'''
    method_name = 'Newton-CG' 
    jac = True
        
#%%---------------------------------------------------------------------------#    
class BFGS(Scipy_Method):
    '''quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)'''
    method_name = 'BFGS'   
    jac = True
#%%---------------------------------------------------------------------------#    
class LBFGSB(Scipy_Method):
    ''' L-BFGS-B algorithm'''
    method_name = 'L-BFGS-B'
    jac = True

#%%---------------------------------------------------------------------------#    
class TNC(Scipy_Method):
    '''truncated Newton (TNC) algorithm'''
    method_name = 'TNC'
    jac = True

#%%---------------------------------------------------------------------------#    
class SLSQP(Scipy_Method):
    '''Sequential Least SQuares Programming (SLSQP)'''
    method_name = 'SLSQP'
    jac = True
    
#%%---------------------------------------------------------------------------# 






  