#!/usr/bin/env python3
"""
Modélisation DCC pour Han, Li, Xia (2017)
"""

import numpy as np


class DCCModeling:
    """Modèle DCC (Dynamic Conditional Correlation) selon Engle (2002)"""
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.95):
        self.alpha = alpha
        self.beta = beta
        self.dynamic_correlations = None
        self.unconditional_corr = None
        
    def estimate_dcc_corrected(self, residuals_matrix: np.ndarray) -> np.ndarray:
        """Estimation DCC selon Engle (2002)"""
        T, n_assets = residuals_matrix.shape
        
        print(f"    Estimation DCC: T={T}, actifs={n_assets}, alpha={self.alpha}, beta={self.beta}")
        
        residuals_clean = residuals_matrix.copy()
        for i in range(n_assets):
            col = residuals_clean[:, i]
            q99 = np.percentile(col[np.isfinite(col)], 99)
            q01 = np.percentile(col[np.isfinite(col)], 1)
            residuals_clean[:, i] = np.clip(col, q01, q99)
        
        Q_bar = np.corrcoef(residuals_clean.T)
        
        eigenvals, eigenvecs = np.linalg.eigh(Q_bar)
        eigenvals = np.maximum(eigenvals, 1e-8)
        Q_bar = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        diag_sqrt = np.sqrt(np.diag(Q_bar))
        Q_bar = Q_bar / np.outer(diag_sqrt, diag_sqrt)
        
        self.unconditional_corr = Q_bar
        
        Q = np.zeros((T, n_assets, n_assets))
        R = np.zeros((T, n_assets, n_assets))
        
        Q[0] = Q_bar.copy()
        
        for t in range(1, T):
            z_prev = residuals_clean[t-1].reshape(-1, 1)
            outer_product = z_prev @ z_prev.T
            
            Q[t] = (1 - self.alpha - self.beta) * Q_bar + \
                   self.alpha * outer_product + \
                   self.beta * Q[t-1]
        
        for t in range(T):
            diag_sqrt_Q = np.sqrt(np.maximum(np.diag(Q[t]), 1e-8))
            D_inv = np.diag(1 / diag_sqrt_Q)
            R[t] = D_inv @ Q[t] @ D_inv
            
            R[t] = np.clip(R[t], -0.999, 0.999)
            np.fill_diagonal(R[t], 1.0)
        
        self.dynamic_correlations = R
        
        print(f"    DCC estimé avec succès")
        print(f"    Corrélation moyenne: {np.mean(R[:, 0, 1]):.4f}")
        print(f"    Corrélation finale: {R[-1, 0, 1]:.4f}")
        
        return R
    
    def convert_to_kendall_tau_dynamic(self, correlations: np.ndarray) -> np.ndarray:
        """Conversion corrélations vers tau de Kendall dynamique selon Eq. (7)"""
        T = correlations.shape[0]
        kendall_tau_dynamic = np.zeros(T)
        
        for t in range(T):
            rho_t = correlations[t, 0, 1]
            tau_t = (2 / np.pi) * np.arcsin(np.clip(rho_t, -0.999, 0.999))
            kendall_tau_dynamic[t] = tau_t
        
        return kendall_tau_dynamic
