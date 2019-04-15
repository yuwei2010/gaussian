# -*- coding: utf-8 -*-
import numpy as np

from .tools import mproperty
from .sample import Samples

#%%---------------------------------------------------------------------------#
class Mean(object):

    #%%-----------------------------------------------------------------------#
    @mproperty
    def name(self):
        
        return self.__class__.__name__
    
    #%%-----------------------------------------------------------------------#
    def __add__(self, other):
        
        return Operator_Sum(self, other)
    
    #%%-----------------------------------------------------------------------#
    def __mul__(self, other):
                    
        return Operator_Product(self, other)
    
    __rmul__ = __mul__ 
    
    #%%-----------------------------------------------------------------------#  
    def __pow__(self, exp):
        
        return Operator_Power(self, exp)
    
#%%---------------------------------------------------------------------------#
class Zero(Mean):
    
    def __init__(self):
        
        self.hyp = np.array([], dtype=float)
        
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):        
        
        return np.zeros((len(x), 1))
    
    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        return np.zeros((len(x), 1))

#%%---------------------------------------------------------------------------#
class One(Mean):
    
    def __init__(self):
        
        self.hyp = np.array([], dtype=float)
        
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):
        
        return np.ones((len(x), 1))
    
    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        return np.zeros((len(x), 1))
    
#%%---------------------------------------------------------------------------#
class Const(Mean):
    '''
    Constant mean function. hyp = [c]

    :param c: constant value for mean
    '''    
    def __init__(self, s=0.):
        
        
        if isinstance(s, Samples):
            
            s = s.y.mean()
            
        self.hyp = np.array([s], dtype=float)

        
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):
        
        return self.hyp * np.ones((len(x), 1))
    
    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        A = np.ones((len(x), 1)) if der == 0 else np.zeros((len(x), 1)) 
        
        return A
    
#%%---------------------------------------------------------------------------#
class Linear(Mean):
    '''
    Linear mean function. self.hyp = alpha_list

    :param D: dimension of training data. Set if you want default alpha, which is 0.5 for each dimension.
    :alpha_list: scalar alpha for each dimension
    '''    
    def __init__(self, s=None):
        
        assert s is not None
        
        if isinstance(s, Samples):
            
            self.hyp = s.y.mean() * np.ones(s.xdim)
        
        elif isinstance(s, int):
            
            self.hyp = np.zeros(s)
        
        elif isinstance(s, (np.ndarray, list, tuple)):
            
            self.hyp = np.asarray(s, dtype=float)
            
        else:
            
            raise ValueError
    
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):
        
        assert len(x.T) == self.hyp.size
        
        return np.dot(x, self.hyp.reshape(-1, 1))

    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):

        if der is None or der > len(x.T):
            
            return np.zeros((len(x), 1))
        
        else:
        
            return x[:, int(der)].reshape(-1, 1)
        
#%%---------------------------------------------------------------------------#
class Operator_Sum(Mean):
    
    def __init__(self, this, other):

        if isinstance(other, (int, float)):
            
            other = Const(other)
        
        self.this = this
        self.other = other
                
        self.hyp = np.r_[this.hyp, other.hyp]
        
        if any(isinstance(mean.name, (list, tuple)) for mean in [this, other]):
            
            if isinstance(this.name, (list, tuple)):
                
                other._name = [other.name]
            else:
                this._name = [this.name]
            
            self._name = list(this.name) + list(other.name)
        else:
            
            self._name = [this.name, other.name]
        
    #%%-----------------------------------------------------------------------#
    @property
    def hyp(self):
        
        return self._hyp
    
    @hyp.setter
    def hyp(self, value):
        
        value = np.asarray(value, dtype=float).ravel()
        
        self._hyp = value
        
        assert value.size == self.hyp.size
                
        self.this.hyp = value[:self.this.hyp.size]
        self.other.hyp = value[self.this.hyp.size:]
        
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):
        
        return self.this.get_mean(x) + self.other.get_mean(x)

    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        assert der < self.hyp.size
        
        if der < self.this.hyp.size:
            
            return self.this.get_mean_der(x, der)
        
        else:
            
            der = der - self.this.hyp.size
            
            return self.other.get_mean_der(x, der)
        
#%%---------------------------------------------------------------------------#
class Operator_Product(Operator_Sum):
    
    def __init__(self, this, other):
                
        super(Operator_Product, self).__init__(this, other)

    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):

        return self.this.get_mean(x) * self.other.get_mean(x)
    
    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        assert der < self.hyp.size
        
        if der < self.this.hyp.size:
            
            return self.this.get_mean_der(x, der) * self.other.get_mean(x)
        
        else:
            
            der = der - self.this.hyp.size
            
            return self.other.get_mean_der(x, der) * self.this.get_mean(x)  
        
#%%---------------------------------------------------------------------------#
class Operator_Power(Operator_Product):
    
    def __init__(self, this, exp):
        
        assert isinstance(exp, (int, float))
        
        super(Operator_Power, self).__init__(this, exp)   
        
    #%%-----------------------------------------------------------------------#
    @property
    def exponent(self):
        
        return np.max(np.r_[1, np.floor(self.other.hyp)])
    
    #%%-----------------------------------------------------------------------#
    def get_mean(self, x):

        return self.this.get_mean(x) ** self.exponent
    
    #%%-----------------------------------------------------------------------#
    def get_mean_der(self, x, der):
        
        if der == 0:
            
            A = self.this.get_mean(x)
            
            return A ** self.exponent * np.log(A)
        
        else:

            return (self.this.get_mean(x) ** (self.exponent - 1) 
                    * self.this.get_mean_der(x, der - 1))
         
#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    m = Const(5)
    
    n = Linear(2)
    
    s = Operator_Sum(m, n)
    p = Operator_Power(n, 6)
    
    print(p.get_mean_der(np.ones(10).reshape(-1, 2), 2))
