#!/usr/bin/env python3
"""
Script principal pour la reproduction de Han, Li, Xia (2017)
"Dynamic robust portfolio selection with copulas"
"""

import os
import numpy as np
from data_pipeline import create_data_files
from han_reproduction import HanReproduction


def main():
    """Fonction principale"""
    print("REPRODUCTION - Han, Li, Xia (2017)")
    print("Dynamic robust portfolio selection with copulas")
    print("="*80)
    
    np.random.seed(42)
    
    # Vérifier si les fichiers de données existent
    log_returns_file = os.path.join('data', 'log_returns.csv')
    
    if not os.path.exists(log_returns_file):
        print("Fichiers de données manquants. Création en cours...")
        success = create_data_files('data')
        if not success:
            print("Échec de la création des fichiers de données")
            return None, None
    
    # Lancer la reproduction
    reproducer = HanReproduction(data_path='data')
    results = reproducer.run_corrected_reproduction()
    
    return reproducer, results


if __name__ == "__main__":
    print("LANCEMENT REPRODUCTION")
    print("="*50)
    
    try:
        reproducer, results = main()
        
        if results is not None:
            print("\n Succès!")
           
        else:
            print("\n ECHEC")
            
    except Exception as e:
        print(f"\nERREUR CRITIQUE: {e}")
        print("\nVERIFICATIONS:")
        print("1. Fichier log_returns.csv présent dans dossier 'data'")
        print("2. Packages: numpy pandas matplotlib scipy arch cvxpy")
        print("3. Permissions lecture/écriture")