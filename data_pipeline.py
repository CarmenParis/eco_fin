#!/usr/bin/env python3
"""
Pipeline de traitement des données pour Han, Li, Xia (2017)
Fusionne les fichiers CSV individuels et calcule les log returns
"""

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """Processeur de données pour actions chinoises"""
    
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.price_data = {}
        self.merged_prices = None
        self.log_returns = None
    
    def load_individual_files(self):
        """Charge les fichiers CSV individuels"""
        csv_files = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.csv') and 'Stock Price History' in file:
                csv_files.append(os.path.join(self.data_folder, file))
        
        if not csv_files:
            return False
        
        for file_path in csv_files:
            try:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                df = pd.read_csv(file_path)
                
                if 'Date' not in df.columns or 'Price' not in df.columns:
                    continue
                
                df = df[["Date", "Price"]].copy()
                
                if df['Price'].dtype == 'object':
                    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
                
                df = df.rename(columns={"Price": f"Price_{filename}"})
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.sort_values("Date").reset_index(drop=True)
                
                self.price_data[filename] = df
                
            except Exception:
                continue
        
        return len(self.price_data) > 0
    
    def merge_data(self):
        """Fusionne les données de prix"""
        if not self.price_data:
            return False
        
        merged_df = list(self.price_data.values())[0].copy()
        
        for df in list(self.price_data.values())[1:]:
            merged_df = pd.merge(merged_df, df, on="Date", how="inner")
        
        merged_df = merged_df.sort_values("Date").reset_index(drop=True)
        self.merged_prices = merged_df
        
        return True
    
    def calculate_log_returns(self):
        """Calcule les rendements logarithmiques"""
        if self.merged_prices is None:
            return False
        
        log_returns = self.merged_prices.copy()
        price_columns = [col for col in log_returns.columns if col.startswith("Price_")]
        
        for col in price_columns:
            log_returns[col] = np.log(log_returns[col]) - np.log(log_returns[col].shift(1))
        
        log_returns = log_returns.dropna().reset_index(drop=True)
        
        log_returns = log_returns.rename(
            columns={col: col.replace("Price_", "LogReturn_") 
                    for col in log_returns.columns if col.startswith("Price_")}
        )
        
        self.log_returns = log_returns
        return True
    
    def save_files(self):
        """Sauvegarde les fichiers générés"""
        if self.merged_prices is None or self.log_returns is None:
            return False, False
        
        merged_path = os.path.join(self.data_folder, "CSI300_subindices.csv")
        self.merged_prices.to_csv(merged_path, index=False)
        
        log_path = os.path.join(self.data_folder, "log_returns.csv")
        self.log_returns.to_csv(log_path, index=False)
        
        return merged_path, log_path
    
    def process_data(self):
        """Lance le traitement complet"""
        if not os.path.exists(self.data_folder):
            print(f"Erreur: Dossier '{self.data_folder}' introuvable")
            return False
        
        if not self.load_individual_files():
            print("Erreur: Aucun fichier de prix trouvé")
            return False
        
        if not self.merge_data():
            print("Erreur: Fusion impossible")
            return False
        
        if not self.calculate_log_returns():
            print("Erreur: Calcul des log returns impossible")
            return False
        
        merged_path, log_path = self.save_files()
        if not merged_path or not log_path:
            print("Erreur: Sauvegarde impossible")
            return False
        
        print(f"Traitement terminé:")
        print(f"  {len(self.log_returns)} observations")
        print(f"  Fichiers générés: CSI300_subindices.csv, log_returns.csv")
        
        return True


def create_data_files(data_folder="data"):
    """Fonction utilitaire pour créer les fichiers de données"""
    processor = DataProcessor(data_folder)
    return processor.process_data()


if __name__ == "__main__":
    processor = DataProcessor()
    success = processor.process_data()
    
    if not success:
        print("Échec du traitement des données")