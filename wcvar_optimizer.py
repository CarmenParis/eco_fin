#!/usr/bin/env python3
"""
Optimisation WCVaR pour Han, Li, Xia (2017)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional
from copula_modeling import CopulaModeling


class WCVaROptimizerCorrected:
    """Optimisation WCVaR selon l'article"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def generate_copula_scenarios_corrected(self, returns_df: pd.DataFrame, 
                                          kendall_tau: float, 
                                          n_scenarios: int = 1000) -> Dict[str, np.ndarray]:
        """Simulation Monte Carlo avec mélanges de copules"""
        scenarios_by_copula = {}
        copula_types = ['gaussian', 'clayton', 'frank', 'gumbel']
        
        print(f"    Génération de {n_scenarios} scénarios avec tau={kendall_tau:.4f}")
        
        for copula_type in copula_types:
            theta = CopulaModeling.kendall_tau_to_theta_corrected(kendall_tau, copula_type)
            
            print(f"      {copula_type}: theta={theta:.4f}")
            
            if copula_type == 'gaussian':
                copula_samples = CopulaModeling.sample_gaussian_copula_corrected(theta, n_scenarios)
            elif copula_type == 'clayton':
                copula_samples = CopulaModeling.sample_clayton_copula_corrected(theta, n_scenarios)
            elif copula_type == 'frank':
                copula_samples = CopulaModeling.sample_frank_copula_corrected(theta, n_scenarios)
            elif copula_type == 'gumbel':
                copula_samples = CopulaModeling.sample_gumbel_copula_corrected(theta, n_scenarios)
            
            scenarios = np.zeros((n_scenarios, 2))
            for i in range(2):
                marginal_data = returns_df.iloc[:, i].dropna().values
                sorted_data = np.sort(marginal_data)
                quantiles = np.clip(copula_samples[:, i], 1e-6, 1-1e-6)
                scenarios[:, i] = np.percentile(sorted_data, quantiles * 100)
            
            scenarios_by_copula[copula_type] = scenarios
            
        return scenarios_by_copula
    
    def solve_wcvar_corrected(self, returns_df: pd.DataFrame, 
                            kendall_tau: float,
                            target_return: Optional[float] = None) -> Dict:
        """Résolution WCVaR avec contraintes selon l'article"""
        n_assets = returns_df.shape[1]
        n_scenarios = 1000
        
        scenarios_dict = self.generate_copula_scenarios_corrected(returns_df, kendall_tau, n_scenarios)
        
        w = cp.Variable(n_assets, nonneg=True)
        alpha = cp.Variable()
        theta = cp.Variable()
        
        constraints = []
        
        constraints.append(cp.sum(w) == 1)
        
        if target_return is not None and target_return > 0:
            expected_returns = returns_df.mean().values
            constraints.append(expected_returns @ w >= target_return)
        
        mixture_weights = {'gaussian': 0.25, 'clayton': 0.25, 'frank': 0.25, 'gumbel': 0.25}
        
        cvar_terms = []
        
        for copula_type, scenarios in scenarios_dict.items():
            weight = mixture_weights[copula_type]
            n_scen = len(scenarios)
            v = cp.Variable(n_scen, nonneg=True)
            
            losses = -scenarios @ w
            
            for k in range(n_scen):
                constraints.append(v[k] >= losses[k] - alpha)
            
            beta = self.confidence_level
            cvar_term = alpha + (1/(1-beta)) * cp.sum(v)/n_scen
            cvar_terms.append(weight * cvar_term)
        
        for cvar_term in cvar_terms:
            constraints.append(cvar_term <= theta)
        
        objective = cp.Minimize(theta)
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            if problem.status == cp.OPTIMAL:
                weights_result = w.value
                if weights_result is not None:
                    weights_result = np.clip(weights_result, 0, 1)
                    weights_result = weights_result / weights_result.sum()
                    
                    return {
                        'weights': weights_result,
                        'wcvar': theta.value,
                        'var': alpha.value,
                        'status': 'optimal'
                    }
        except Exception as e:
            print(f"    Erreur optimisation: {e}")
        
        return {
            'weights': np.array([0.5, 0.5]),
            'wcvar': np.nan,
            'var': np.nan,
            'status': 'failed'
        }
