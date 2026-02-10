"""
Archivo: main.py
Descripción: Script principal para ejecutar el pipeline del TFM.
             1. Carga datos.
             2. Calcula error de la física (ASTM E900).
             3. (Futuro) Entrena la red neuronal.
"""
from src.baseline_scripts.sin_ai import calcular_rmse_sin_ai

def main():
    calcular_rmse_sin_ai()

if __name__ == "__main__":
    main()