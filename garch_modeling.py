#!/usr/bin/env python3
"""
Modélisation GARCH pour Han, Li, Xia (2017)
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict


class GARCHModeling:
    """Modélisation GARCH(1,1) avec distribution t de Student"""
    
    def __init__(self):
        self.fitted_models = {}
        self.standardized_residuals = {}
        self.conditional_volatility = {}
        self.garch_params = {}
        
    def fit_garch_t_student(self, returns: pd.Series, name: str) -> Dict:
        """Ajustement GARCH(1,1) avec distribution t de Student"""
        print(f"Ajustement GARCH(1,1)-t pour {name}...")
        
        try:
            clean_returns = returns.dropna()
            if len(clean_returns) < 100:
                raise ValueError(f"Pas assez de données pour {name}")
            
            model = arch_model(
                clean_returns * 100,
                vol='GARCH', 
                p=1, q=1, 
                dist='t',
                rescale=False
            )
            
            fitted = model.fit(disp='off', show_warning=False, options={'maxiter': 2000})
            
            conditional_vol = fitted.conditional_volatility / 100
            residuals = fitted.resid / 100
            standardized_residuals = residuals / conditional_vol
            
            self.fitted_models[name] = fitted
            self.conditional_volatility[name] = conditional_vol
            self.standardized_residuals[name] = standardized_residuals
            
            params = fitted.params
            omega = params['omega'] / 10000
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            nu = params['nu'] if 'nu' in params else np.nan
            
            self.garch_params[name] = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'nu': nu,
                'persistence': alpha + beta,
                'loglikelihood': fitted.loglikelihood,
                'aic': fitted.aic,
                'bic': fitted.bic
            }
            
            print(f"    omega={omega:.8f}, alpha={alpha:.4f}, beta={beta:.4f}, nu={nu:.2f}, alpha+beta={alpha+beta:.4f}")
            
            return {
                'fitted_model': fitted,
                'conditional_volatility': conditional_vol,
                'standardized_residuals': standardized_residuals,
                'params': self.garch_params[name],
                'success': True
            }
            
        except Exception as e:
            print(f"    Erreur GARCH pour {name}: {e}")
            
            vol_empirique = returns.rolling(window=30, min_periods=10).std().fillna(returns.std())
            residuals_fallback = returns / vol_empirique
            
            return {
                'fitted_model': None,
                'conditional_volatility': vol_empirique,
                'standardized_residuals': residuals_fallback,
                'params': {'omega': np.nan, 'alpha': np.nan, 'beta': np.nan, 'nu': np.nan, 'persistence': np.nan},
                'success': False
            }
