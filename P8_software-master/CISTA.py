# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:41:53 2019

@author: Tobias
"""

import numpy as np

class CISTA:
    """
    Class for solving problems of the form:
        
        min 0.5*||Ax-y||_2^2 + \lambda*||x||_1 for x in C^n
    
    where: 
        A in C^{m\times n} is the measurement matrix (m << n in general)
        y in C^m is the measured data
        x in C^n unknown sparse vector
        lambda > 0 is the reguliser on x (how sparse should solution be)
    
    Functionality:
         - initialise parameters (see __init__)
         - Solve optimization problem using CISTA (see optimize)
         - Analyse convergence of optimization (see optimize agian)
    
    """
    def __init__(self, A, y, lamb=0.5, complex=True):
        """
        Initialise parameters CISTA algorithm.
        
        Paramters
        ---------
        A : (m,n) size numpy array
            Mearurement matrix
        y : (m,p) size numpy array
            Measured data. For p > 1 it is a matrix and the solution will be
            a (n,p) size matrix as well
        lamb : positive float, optional
            Weight on L_1 reguliser. Default to 0.5
        complex : bool, optional
            Weather to compute CISTA (True) or ISTA (False). Defaults to True
        """
        
        self.A = A
        self.y = y
        self.lamb = lamb
        self.m, self.n = A.shape
        if len(y.shape) > 1:
            self.p = y.shape[1]
        else:
            self.p = 1
        
        
        #Run som sanity checks
        assert(isinstance(A,np.ndarray))
        assert(isinstance(y,np.ndarray))
        assert(self.m == self.y.shape[0])
        assert(self.lamb > 0)
        if not complex:
            assert(not np.iscomplexobj(A))
            assert(not np.iscomplexobj(y))
        

        if complex:
            self.A_backward = self._Hermitian(self.A)
        else:
            self.A_backward = self.A.T
        
        #get M as A^H*A which is used more than once 
        M = self.A_backward.dot(self.A)
        
        self.alpha = 1/np.linalg.norm(A,ord = 2)**2 

        
        #Compute matricies involved in gradient step in CISTA
        self.Phi = np.eye(self.n) - self.alpha*M
        self.phi_y = self.alpha*self.A_backward.dot(self.y)

        
    def optimize(self, x0='zeros', x_sol=None, return_mode='all',
                 MAX_ITER=300, tol=1e-4, patience=5, verbose=0):
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
        MAX_ITER : positive int
            Maximum number of iterations of CISTA algorithm.
        tol : float
            If relative change in x_k is below tol for 'patience' iterations,
            algorithm will stop.
        patience : positve int or str
            See tol. Then patience is set to 'inf', only MAX_ITER is
            stop criteria
        
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
        if isinstance(x0,str):
            if x0 == 'zeros':
                if self.p == 1:
                    x0 = np.zeros(self.n)
                else:
                    x0 = np.zeros([self.n,self.p])
            
        #Run some sanity checks
        else: 
            assert(isinstance(x0,np.ndarray))
            assert(x0.shape[0]  == self.n)
            if self.p > 1:
                 assert(x0.shape[1] == self.p)
        assert(isinstance(MAX_ITER,int) and MAX_ITER > 0)
        assert((isinstance(patience,int) and patience > 0) or patience == 'inf')
        assert(return_mode in ('all','sol','obj'))
        
        #Setup optimization
        x = x0
        #If solution is given, both objective cost and NMSE will be tracked
        if x_sol is not None:
            hist = np.zeros([MAX_ITER+1,2])
            x_MS = np.linalg.norm(x_sol)**2
            hist_update = lambda x: [self._objective(x),self._NMSE(x_sol,x_MS,x)]
        #Else track only onjective cost. 
        else:
            hist = np.zeros(MAX_ITER+1)
            hist_update = lambda x: self._objective(x)
        
        #Set initial values
        hist[0] = hist_update(x)
        
        #Setup tolerence check
        check_patience = True
        if isinstance(patience,str):
            check_patience = False
            patience = 5 #needs to be int for later comparison but is not used
        n_patience = 0
        #Perform CISTA
        
        for k in range(MAX_ITER):
            x_km1 = x #Save x_{k-1} for later comparison
            z = self.Phi.dot(x) + self.phi_y #Gradient Step
            x = self._threshold(z) #Proximal step
            obj = self._objective(x)
            hist[k+1] = hist_update(x)
            
            #Test if relative change is too small
            if check_patience:
                if self._relative_change_tol(x,x_km1,tol):
                    n_patience = 0 #reset patience when relative change is above tol
                else:
                    n_patience += 1
                #Stop when patience is met
                if n_patience == patience: 
                    hist = hist[:k+1] #remove zeros in the end of array
                    if verbose:
                        print("Stopped after {} iterations. Relative change was below tol for {} iterations".format(k,patience))
                    break
        #Print stop message when MAX_ITER is met
        if verbose:
            if k + 1 == MAX_ITER and n_patience < patience:
                print("Stopped after {} iterations. MAX_ITER met.".format(k+1))
        
        if return_mode == 'all':
            return({'xhat': x, 'obj': obj, 'history': hist})
        elif return_mode == 'sol':
            return(x)
        elif return_mode == 'obj':
            return(obj)
    
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
    
    def _Hermitian(self, X):
        """
        Computes hermitian conjugate of matrix X: conj(X)^T
        """
        return(np.conj(X).T)
        
    def _threshold(self, z):
        """
        Threshold function. This implementation avoids dividing by zero
        accidentially. 
        """
        a = np.maximum(np.abs(z), self.alpha*self.lamb)
        x = (1 - (self.alpha*self.lamb)/a)*z
        return(x)
    
    def _NMSE(self, x, x_MS, xhat):
        """
        Compute the normalized MSE: ||x-xhat||_2^2/||x||_2^2
        The ||x||_2^2 is given as argument to avoid computing it more than
        once
        """
        return np.linalg.norm(x-xhat)**2/x_MS

    def _relative_change_tol(self, x2, x1, tol):
        """
        Measure of change in x2 relative to x1. ||x||_2 is multiplied out on 
        both sides to avoid dividing by zero
        """
        return(np.linalg.norm(x2 - x1) > tol*np.linalg.norm(x1))
        
