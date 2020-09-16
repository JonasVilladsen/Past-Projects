#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:41:44 2019

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

class NB_channel_simu():
    """
    Simulates a narrow band channel given the channel parameters.
    
    Example:
        
        Ex = NB_channel_simu(20,4,64,64)
        Y,S,W = Ex.simulate()
    """
    def __init__(self, N, L, N_t, N_r, theta_distribution="uniform",
                 fading="Rayleigh", sigma=1, profile="exponential", a=1,
                 delay=0.3, modulation='QPSK', SNR=0, save_dict=False,
                 verbose=1):
        
        """
        Parameters
        ----------
        N : integer
            number describing the number of channel uses
            
        L : integer
            number describing the number of clusters
            
        N_t : integer
            number describing the number of transmitters
            
        N_r : integer
            number describing the number of receivers
            
        theta_distribution : string
            describes how theta_N_t and theta_N_r should be distributed
            
        fading : string
            describes how to construct beta_l
            
        sigma : float (only relevant if fading == Rayleigh)
            number defining the std of beta
            
        a : float (Only relevant if fading == ExpFading)
            number scaling the power delay profile
                            S(\tau) = 1/a * exp(-\tau/a)
        
        delay : float
            Characterises the exponential distribution of tau. If choosen too 
            high the later beta_l will be weighted down significantly.
            
        modulation : string
            Defines which set S_t should be sampled from
        
        SNR : float
            Signal to noise ratio. Not in dB!
        
        save_dict : boolean
            Determines wether to return a dictonary or not
                save_dict = True saves a dictonary containing the relevant info
        """
        if verbose: 
            if N > N_t:
                print('This configuration will have a tall S matrix, choose N << N_t')
            if L > N_t*N_r:
                print('Attention : this channel will unlikely be sparse. Choose L << N_t*N_r')
        
        self.N = N
        self.L = L
        self.N_r = N_r
        self.N_t = N_t

        self.theta_distribution = theta_distribution
        self.fading = fading
        self.modulation = modulation
        self.SNR = SNR
        
        self.save_dict = save_dict
        
        if self.fading == 'ExpFading':
            self.profile = profile
            self.a = a # brugt i konstruktionen af sigma
            self.delay = delay
            self.sigma_l = self._construct_sigma()
            self.beta = self.CN_beta(self.sigma_l)
            self.theta_N_t, self.theta_N_r = self._construct_theta()
            self.f_N_t, self.f_N_r = self.f_response()
            self.F_N_t, self.F_N_r, self.F_N = self._construct_F_transform()
            self.G = self._construct_G()
            
        if self.fading == "Rayleigh":
            self.profile = None
            self.sigma = sigma
            self.sigma_l = np.repeat(self.sigma, L)
            self.beta = self.CN_beta(self.sigma_l)
            self.theta_N_t, self.theta_N_r = self._construct_theta()
            self.f_N_t, self.f_N_r = self.f_response()
            self.F_N_t, self.F_N_r, self.F_N = self._construct_F_transform()
            self.G = self._construct_G()

        elif self.fading != 'Rayleigh' and self.fading != 'ExpFading':
            raise(ValueError('Not implemented. Choose Rayleigh or ExpFading.'))
        
    def __str__(self):
        s = 'The number of channel uses is N = {:d}. \n' + \
        'The number of clusters is L = {:d}. \n' + \
        'The number of receivers is N_r = {:d}. \n' + \
        'The number of transmitters is N_t = {:d}. \n' + \
        'Theta is distributed as {:s}. \n' + \
        'The fading used is {:s} with {:s} PDP. \n' + \
        'The modulation used is {:s}.'
        
        if self.profile == "exponential":
            return s.format(self.N, self.L, self.N_r, self.N_t,
                            self.theta_distribution, self.fading, self.profile,
                            self.modulation)
        else:
            return s.format(self.N, self.L, self.N_r, self.N_t,
                            self.theta_distribution, self.fading, "no",
                            self.modulation)
        
    def CN_beta(self, sigma_l):
        """
        Samples the complex gain from a CN distribution
        """
        beta = np.zeros(shape=(self.L), dtype='complex')
        for l in range(self.L):
            beta[l] = np.random.normal(0, sigma_l[l]/2) + \
                        1j*np.random.normal(0, sigma_l[l]/2)
        return beta
    
    def _H(self, X):
        """
        Returns the adjungated matrix
        """
        return np.conjugate(X).T
    
    def _construct_sigma(self):
        """
        Constructs sigma_l using the accumulated sum of an exponential 
        distribution and applying the exponential PDP
        """
        
        self.tau = np.cumsum(np.random.exponential(self.delay, self.L))
        if self.profile == "exponential":
            return (1/self.a)*np.exp(-(self.tau/self.a))
    
    def _construct_theta(self):
        """
        Constructs theta_N_t and theta_N_r as either
            L samples between 0 and 2pi/N_r or 2pi/N_t uniformly spaced by L.
            L samples between 0 and 2pi sampled with an uniform distribution.
        """
        if self.theta_distribution == "uniform":
            theta_N_t = np.random.uniform(0, 2*np.pi, self.L)
            theta_N_r = np.random.uniform(0, 2*np.pi, self.L)
        else:
            theta_N_t = np.array([2*np.pi*i/self.N_t for i in range(1, self.L+1)])
            theta_N_r = np.array([2*np.pi*i/self.N_r for i in range(1, self.L+1)])
        return theta_N_t, theta_N_r
    
    def f_response(self):
        """
        Calculates the transmit and recieve arrays responses
        """
        f_N_t = np.zeros((self.N_t,self.L), dtype="complex")
        f_N_r = np.zeros((self.N_r,self.L), dtype="complex")
        for i in range(self.L):
            f_N_t[:,i] = (1/np.sqrt(self.N_t))*np.exp(-1j*np.arange(self.N_t)*\
                 self.theta_N_t[i])
            f_N_r[:,i] = (1/np.sqrt(self.N_r))*np.exp(-1j*np.arange(self.N_r)*\
                 self.theta_N_r[i])
        return f_N_t, f_N_r
    
    def _construct_F_transform(self):
        """
        Calculates the Fourier transform matrix for N, N_r and N_t 
        """
        F_N_t = np.zeros((self.N_t,self.N_t), dtype="complex")
        F_N_r = np.zeros((self.N_r,self.N_r), dtype="complex")
        F_N = np.zeros((self.N,self.N), dtype="complex")
        for i in range(self.N_t):
            for j in range(self.N_t):
                F_N_t[i,j] = np.exp(-2*np.pi*1j*i*j/self.N_t)
                
        for i in range(self.N_r):
            for j in range(self.N_r):
                F_N_r[i,j] = np.exp(-2*np.pi*1j*i*j/self.N_r)
                
        for i in range(self.N):
            for j in range(self.N):
                F_N[i,j] = np.exp(-2*np.pi*1j*i*j/self.N)
        return(F_N_t/np.sqrt(self.N_t), F_N_r/np.sqrt(self.N_r), F_N/np.sqrt(self.N))
        
    def _construct_G(self):
        G = np.zeros((self.N_t, self.N_r), dtype="complex")
        for l in range(self.L):
            z_N_t = self._H(self.F_N_t).dot(self.f_N_t[:,l])
            z_N_r = self._H(self.f_N_r[:,l]).dot(self.F_N_r)
            G += self.beta[l]*np.outer(z_N_t, z_N_r)
        return G

    def plot_G(self, save=False, name='G'):
        """
        Nicely plots the channel matrix
        """
        plt.figure()
        G = self._construct_G()
        plt.title(r'$\vert G_0 \vert, L = {:d}$'.format(self.L))
        plt.imshow(abs(G), origin='lower')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.ylabel(r'$N_t$', rotation=1)
        plt.xlabel(r'$N_r$')
        if save == True:
            plt.savefig(name + '.png', dpi=500, bbox_inches='tight')
        plt.show()

    def _construct_S(self):
        """
        Samples the modulated signal S_t
        """
        if self.modulation == 'BPSK':
            Sample_set = np.array([1,-1], dtype="complex")
        elif self.modulation == 'QPSK':
            Sample_set = np.array([1,-1,1j,-1j], dtype="complex")
        else:
            raise(ValueError('Choose either BPSK or QPSK as modulation.'))
        return np.random.choice(Sample_set, size=(self.N,self.N_t))
    
    def _construct_W(self, S, G):
        """
        Returns W with the correct SNR
        """
        power_SG = np.linalg.norm(np.dot(S,G),'fro')**2/(self.N*self.N_r) # expected value of the Frobenius norm
        sigma_W = power_SG*10**(-self.SNR/10)/power_SG.size # SNR = 10*log(Power_SG/powerNoise)
        W_re = np.random.normal(0, sigma_W/2, self.N*self.N_r)
        W_im = np.random.normal(0, sigma_W/2, self.N*self.N_r)
        W = (W_re + 1j*W_im).reshape(self.N, self.N_r)
        return W
    
    def _construct_dict(self, Y, S, W):
        NB_channel_dict = {}
        NB_channel_dict['Y'] = Y
        NB_channel_dict['S'] = S
        NB_channel_dict['W'] = W
        NB_channel_dict['G'] = self.G       
        return NB_channel_dict
        
    def save_sim(self, info_dict):
        filename = 'Nt{:d}_Nr{:d}_'.format(self.N_t, self.N_r) + 'H' + \
                str(datetime.datetime.now())[0:13] + 'M' + \
                str(datetime.datetime.now())[14:16] + '.pickle'
        file = open(filename, 'wb')
        pickle.dump(info_dict, file)
        file.close()

    def simulate(self):
        """
        Simulates the narrow band channel
        """
        
        S_save = self._construct_S()
        G = self.G
        Y_temp = S_save.dot(G)
        barW_save = self.F_N.dot(self._construct_W(S_save, G)).dot(self.F_N_r)
        Y = Y_temp + barW_save
        
        if self.save_dict == False:
            return self._construct_dict(Y, S_save, barW_save)
        else:
            return self.save_sim(self._construct_dict(Y, S_save, barW_save))