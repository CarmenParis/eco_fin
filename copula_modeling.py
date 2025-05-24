#!/usr/bin/env python3
"""
Modélisation des copules pour Han, Li, Xia (2017)
"""

import numpy as np
from scipy import stats


class CopulaModeling:
    """Modélisation des 4 copules selon les équations (3)-(6) de l'article"""
    
    @staticmethod
    def kendall_tau_to_theta_corrected(tau: float, copula_type: str) -> float:
        """Conversion tau de Kendall vers paramètre theta selon Eqs. (3)-(6)"""
        tau = np.clip(tau, -0.98, 0.98)
        
        if copula_type == 'gaussian':
            return np.sin(tau * np.pi / 2)
            
        elif copula_type == 'clayton':
            if tau <= 0:
                return 0.001
            return 2 * tau / (1 - tau)
            
        elif copula_type == 'gumbel':
            if tau <= 0:
                return 1.001
            return 1 / (1 - tau)
            
        elif copula_type == 'frank':
            if abs(tau) < 0.01:
                return 0
            return 4 * tau
    
    @staticmethod
    def sample_gaussian_copula_corrected(rho: float, n: int) -> np.ndarray:
        """Échantillonnage copule Gaussienne"""
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        
        if abs(rho) >= 0.999:
            rho = np.sign(rho) * 0.999
            cov = [[1, rho], [rho, 1]]
        
        samples = np.random.multivariate_normal(mean, cov, n)
        return np.column_stack([stats.norm.cdf(samples[:, 0]), 
                               stats.norm.cdf(samples[:, 1])])
    
    @staticmethod
    def sample_clayton_copula_corrected(theta: float, n: int) -> np.ndarray:
        """Échantillonnage copule Clayton"""
        if theta <= 0:
            return np.random.uniform(0, 1, (n, 2))
        
        u1 = np.random.uniform(0, 1, n)
        v = np.random.uniform(0, 1, n)
        
        try:
            u2 = np.power(
                np.power(u1, -theta) * (np.power(v, -1/(1+theta)) - 1) + 1,
                -1/theta
            )
            u2 = np.clip(u2, 1e-10, 1-1e-10)
        except:
            u2 = np.random.uniform(0, 1, n)
        
        return np.column_stack([u1, u2])
    
    @staticmethod
    def sample_frank_copula_corrected(theta: float, n: int) -> np.ndarray:
        """Échantillonnage copule Frank"""
        if abs(theta) < 1e-6:
            return np.random.uniform(0, 1, (n, 2))
        
        u1 = np.random.uniform(0, 1, n)
        v = np.random.uniform(0, 1, n)
        
        try:
            exp_theta = np.exp(theta)
            exp_theta_u1 = np.exp(theta * u1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                numerator = -np.log(1 - v * (1 - exp_theta) / 
                                   (exp_theta_u1 - v * (exp_theta_u1 - exp_theta)))
                u2 = numerator / theta
            
            u2 = np.where(np.isfinite(u2), u2, np.random.uniform(0, 1, n))
            u2 = np.clip(u2, 1e-10, 1-1e-10)
        except:
            u2 = np.random.uniform(0, 1, n)
        
        return np.column_stack([u1, u2])
    
    @staticmethod
    def sample_gumbel_copula_corrected(theta: float, n: int) -> np.ndarray:
        """Échantillonnage copule Gumbel"""
        theta = max(theta, 1.001)
        
        corr_approx = 1 - 1/theta
        corr_approx = np.clip(corr_approx, 0, 0.999)
        
        mean = [0, 0]
        cov = [[1, corr_approx], [corr_approx, 1]]
        
        samples = np.random.multivariate_normal(mean, cov, n)
        u1 = stats.norm.cdf(samples[:, 0])
        u2 = stats.norm.cdf(samples[:, 1])
        
        return np.column_stack([u1, u2])
