# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:41:53 2019

@author: Martin Kamp Dalgaard
"""

import numpy as np

class SCISTA:
    """
    Class for solving problems of the form:
    
        min 0.5*||Ax-y||_2^2 + \lambda*||x||_1 for x in C^n
    
    where:
        A in C^{m\times n} is the measurement matrix (m << n in general)
        y in C^m is the measured data
        x in C^n unknown sparse vector
        lambda > 0 is the regularizer on x (how sparse the solution should be)
    
    Functionality:
         - initialise parameters (see __init__)
         - Solve optimization problem using CISTA (see optimize)
         - Analyse convergence of optimization (see optimize again)
    """
    def __init__(self, A, y, c, xmax, T=200, lamb=0.5,
                 pre_evaluated_b_spline=None, pre_eval_res=int(1e6)):
        """
        Initialise parameters for SCISTA algorithm.
        
        Paramters
        ---------
        A : (m,n) size numpy array
            Measurement matrix.
        y : (m,p) size numpy array
            Measured data. For p > 1 it is a matrix and the solution will be
            a (n,p) size matrix as well.
        c : (2K+1) size array
            weights for sum in spline evaluation.
        xmax : positive flaot
            Maximum value expected for input to Ax (element-wise).
        T : int
            Number of iterations in L_ISTA algorithm.
        lamb : positive float, optional
            Weight on L_1 reguliser. Default to 0.5
        complex : bool, optional
            Wether to compute SCISTA (True) or SISTA (False). Defaults to True.
        """
        
        if len(y.shape) > 1:
            self.p = y.shape[1]
        else:
            self.p = 1
        
        self.m, self.n = A.shape
        
        self.y = y
        self.A = A
        
        self.lamb = lamb
        self.c = c
        self.K = int((c.shape[0]-1)/2)
        self.xmax = xmax
        self.delta = (self.xmax*1.1)/(2+self.K)
        self.T = T
        
        #Run som sanity checks
        assert(isinstance(A, np.ndarray))
        assert(isinstance(y, np.ndarray))
        #assert(self.m == self.y.shape[0])
        assert(self.lamb > 0)
        
        #Now calculate some parameters based on A and y
        self.update_parameters(y, A)
        
        #Pre evaluate cubic spline
        self.eval_xmax = 2.1 
        self.pre_eval_res = pre_eval_res
        if pre_evaluated_b_spline is None:
            assert(self.eval_xmax == 2.1)
            assert(self.pre_eval_res == int(1e6)) #the resolution saved evaluation is
            self.cubic_pre_evaluated = np.load('coef/cubic_evaluated_new.npy')
        else: #evaluate again
            self._pre_evaluate_cubic_spline()
    
    def update_parameters(self, y, A):
        A_backward = A.T
        
        n = A.shape[1]
        
        self.alpha = 1/np.linalg.norm(A, ord=2)**2
        
        #Compute matricies involved in gradient step in CISTA
        self.Phi = np.eye(n) - self.alpha*A_backward.dot(A)
        self.phi_y = self.alpha*A_backward.dot(y)
        
        #Set k idx
        if self.p == 1:
            #Psi with z as vector
            self.k_idx = np.tile(np.arange(-self.K, self.K+1), n)\
                                        .reshape(n, 2*self.K+1)    
        else:
            #Tobias: Trust me it works like this :)
            self.k_idx = np.tile(np.arange(-self.K, self.K+1), n*self.p)\
                                    .reshape(self.p, n, 2*self.K+1)
    
    def optimize(self, x0='zeros', x_sol=None, return_mode='sol'):
        """
        Given x0, optimize using CISTA which is a two step optimization 
        algorithm with a gradient step an a proximal step. 
        
        Parameters
        ----------
        x0 : numpy array of size (n,p) or str
            Initial x. defaults to the zero vector/matrix of size (n,p)
        x_sol : numpy array of size (n,p) or None
            If given, the NMSE between x_k and x_sol is tracked at each 
            iteration
        return_mode : str
            all: Return all the above mentioned as d
            sol: Return only sol as float
            obj: return only objective value as float          
            latent: return last x and history of latent variables
            
        Returns
        -------
        Dictionary containing the following:
        xhat : numpy array of size (n,p)
            The found solution x
        obj : float
            Value for the objective function at sol
        history : numpy array
            Objective function for each iteration inclusing for x0 and sol
        """
        
        if isinstance(x0, str):
            if x0 == 'zeros':
                if self.p == 1:
                    x0 = np.zeros(self.n)
                else:
                    x0 = np.zeros([self.n, self.p])
        
        #Run some sanity checks
        else:
            assert(isinstance(x0, np.ndarray))
            assert(x0.shape[0] == self.n)
            if self.p > 1:
                 assert(x0.shape[1] == self.p)
        assert(return_mode in ('sol', 'latent', 'all'))
        
        #Setup optimization
        x = x0
        
        #If solution is given, both objective cost and NMSE will be tracked
        if x_sol is not None:
            hist = np.zeros([self.T, 2])
            x_MS = np.linalg.norm(x_sol)**2
            hist_update = lambda x: [self._objective(x), self._NMSE(x_sol, x_MS, x)]
        #Else track only objective cost. 
        else:
            hist = np.zeros(self.T)
            hist_update = lambda x: self._objective(x)
        
        #Set initial values
        if self.p > 1:
            z_hist = np.zeros((self.T, self.n, self.p))
        else:
            z_hist = np.zeros((self.T, self.n))
        
        #Perform spline_CISTA
        for k in range(self.T):
            
            #Gradient Step
            z = self.Phi.dot(x) + self.phi_y
            
            #Proximal step
            Psi = self._get_Psi(z)
            x = Psi.dot(self.c)
            
            #Transpose needed for higher dimension
            if self.p > 1:
                x = x.T
            
            z_hist[k] = z
            
            hist[k] = hist_update(x)
        
        if return_mode == 'sol':
            return(x)
        elif return_mode == 'latent':
            return({'xhat': x, 'z_hist': z_hist})
        elif return_mode == 'all':
            return({'xhat': x, 'z_hist': z_hist, 'history': hist})
    
    def _Hermitian(self, X):
        """
        Computes hermitian conjugate of matrix X: conj(X)^T
        """
        return(np.conj(X).T)
    
    def _objective(self, x):
        """
        Computes the objective
        
        The 2 norm or frobeneous norm will be computed in the first norm 
        depending on if x,y are vectors or matricies. 
        
        x is reshapen in the second norm to get the desired result when 
        x is matrix
        """
        return(0.5*np.linalg.norm(self.A.dot(x) - self.y)**2 +
               self.lamb*np.linalg.norm(x.reshape(self.n*self.p), ord=1))

    def _NMSE(self, x, x_MS, xhat):
        """
        Compute the normalized MSE: ||x-xhat||_2^2/||x||_2^2
        The ||x||_2^2 is given as argument to avoid computing it more than
        once
        """
        return np.linalg.norm(x-xhat)**2/x_MS
    
    def _threshold(self, z):
        """
        Threshold function. This implementation avoids dividing by zero
        accidentially. 
        """
        a = np.maximum(np.abs(z), self.alpha*self.lamb)
        x = (1 - (self.alpha*self.lamb)/a)*z
        return(x)
    
    def _cubic_spline(self, z):
        """
        Evaluates cubic spline with z as numpy array or int.
        The dimensions of the input and output are equal.
        """
        z_abs = np.abs(z)
        c1 = z_abs <= 1
        c2 = z_abs <= 2
        c3 = np.invert(c1)
        
        a = 2/3 - z_abs**2 + z_abs**3/2
        b = (1/6)*(2-z_abs)**3
        return(c1*a + c2*c3*b)
        
    def _pre_evaluate_cubic_spline(self):
        self.cubic_pre_evaluated = self._cubic_spline(
                np.linspace(-self.eval_xmax, self.eval_xmax, self.pre_eval_res))
    
    def _cubic_spline_fast_eval(self, z):
        #all values not in the range [-2,2] are set to 2 and evaluated to zero
        z[(z < - 2) | (z > 2)] = 2
        index = ((z+self.eval_xmax)*self.pre_eval_res/(2*self.eval_xmax)).astype('int')
        return(self.cubic_pre_evaluated[np.array(index)])
    
    def _get_Psi(self, z, n=None, p=None):
        """
        Evaluates cubic spline at dynamic range and returns Psi.
        """
        #n can be set to different values but defaults to self.n
        if n is None:
            n = self.n
        #Same for p
        if p is None:
            p = self.p
        
        if p == 1:
            #Psi with z as vector
            Z = np.repeat(z, 2*self.K+1).reshape(n, 2*self.K+1)
        
        else:
            #Tobias: Trust me it works like this :)
            Z = np.repeat(z.T, 2*self.K+1).T.reshape(p, n, 2*self.K+1)
        
        idx = Z/self.delta - self.k_idx
        Psi = self._cubic_spline_fast_eval(idx)
        return(Psi)

class fast_cubic_evaluate:
    
    def get_pre_evaluated_list(self, pre_eval_res=int(1e6)):
        self.pre_eval_res = pre_eval_res
        self.eval_xmax = 2.1
        self.cubic_pre_evaluated = self._cubic_spline(
                np.linspace(-self.eval_xmax, self.eval_xmax, self.pre_eval_res))
        return(self.cubic_pre_evaluated)

    def __call__(self, z):
         #all values not in the range [-2,2] are set to 2 and evaluated to zero
        z[(z < - 2) | (z > 2)] = 2
        index = ((z+self.eval_xmax)*self.pre_eval_res/(2*self.eval_xmax)).astype('int')
        return(self.cubic_pre_evaluated[np.array(index)])

    def _cubic_spline(self, z):
        """
        Evaluates cubic spline with z as numpy array or int.
        The dimensions of the input and output are equal.
        """
        z_abs = np.abs(z)
        c1 = z_abs <= 1
        c2 = z_abs <= 2
        c3 = np.invert(c1)
        
        a = 2/3 - z_abs**2 + z_abs**3/2
        b = (1/6)*(2-z_abs)**3
        return(c1*a + c2*c3*b)
                