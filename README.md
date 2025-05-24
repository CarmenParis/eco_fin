# Projet - Reproduction du papier Dynamic Robust Portfolio Selection with Copulas

Ce projet reproduit l’approche proposée par Han, Li et Xia (2017) dans leur article *"Dynamic Robust Portfolio Selection with Copulas"*.  
Il combine la modélisation robuste du risque via le Worst-Case CVaR (WCVaR) et la dépendance dynamique entre actifs grâce aux copules.

## Objectif
Optimiser un portefeuille d'actifs en prenant en compte :
- La modélisation de la volatilité via GARCH(1,1)
- La dépendance entre actifs via des copules paramétriques (Gaussienne, Clayton, Gumbel, Frank)
- La dépendance **dynamique** via le modèle DCC
- La minimisation du risque extrême avec le **Worst-Case CVaR** (WCVaR)
- L’évaluation de plusieurs stratégies : Nonrobust, Scopula, GARCH-Copula, MixCopula (DCC)

## Structure du projet

PROJET_ECONO_FIN/
│
├── eco_fin/                 
│   ├── __pycache__/
│
├── data/                    
├── resultats/               
│
├── copula_modeling.py        
├── dcc_modeling.py           
├── garch_modeling.py         
├── wcvar_optimizer.py     
├── data_pipeline.py          
├── han_reproduction.py       
├── main.py                   
├── README.md                 

## Installation 

pip install numpy pandas matplotlib seaborn
pip install arch cvxpy

## Exécution

Pour lancer l'ensemble de la pipline depuis le terminal : python main.py
