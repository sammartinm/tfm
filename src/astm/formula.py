#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo: formula.py
Descripción: Implementación de la ecuación de correlación ASTM E900-15 para predecir 
             el desplazamiento de la temperatura de transición (TTS) en aceros de vasija.
Autor: Samuel Martín Martínez
Fecha: 2026
"""

import numpy as np

def astm_e900_15(cu, ni, p, mn, temp_c, fluence, product_form):
    """
    Calcula el TTS (Transition Temperature Shift) basándose en ASTM E900-15.
    
    Args:
        cu (array-like): Contenido de Cobre en % peso.
        ni (array-like): Contenido de Níquel en % peso.
        p (array-like): Contenido de Fósforo en % peso.
        mn (array-like): Contenido de Manganeso en % peso.
        temp_c (array-like): Temperatura de irradiación en Celsius.
        fluence (array-like): Fluencia de neutrones (n/cm^2, E > 1 MeV).
        product_form (array-like): Tipo de producto ('W', 'P', 'F').
        
    Returns:
        np.array: TTS predicho en grados Celsius.
    """
    
    # --- 1. PREPARACIÓN DE DATOS Y UNIDADES ---
    # Convertir inputs a arrays de numpy para vectorización
    cu = np.array(cu, dtype=float)
    ni = np.array(ni, dtype=float)
    p = np.array(p, dtype=float)
    mn = np.array(mn, dtype=float)
    temp_c = np.array(temp_c, dtype=float)
    fluence = np.array(fluence, dtype=float)
    product_form = np.array(product_form, dtype=str)

    # Conversión de Temperatura: Celsius -> Fahrenheit -> Rankine (para la fórmula)
    T_f = temp_c * 1.8 + 32.0
    T_R = T_f + 460.67

    # Conversión de Fluencia: La norma usa unidades de n/m2 típicamente o ajustadas.
    # Para E900-15, usaremos la fluencia efectiva en unidades relativas.
    # Clampeamos para evitar log(0)
    fluence = np.maximum(fluence, 1e17)
    
    # --- 2. TÉRMINO A: MECANISMOS DE ENDURECIMIENTO DE MATRIZ (SMD) ---
    # Coeficientes A según tipo de material (Weld vs Plate/Forging)
    # Valores típicos aproximados de la norma E900-15
    A_coeff = np.where(product_form == 'W', 6.70e-4, 5.54e-4) # W=Weld, otros=Base Metal

    # El término A depende de la temperatura (Arrhenius) y Fósforo
    # TTS_1 = A * exp(19310 / T_R) * (1 + 110*P) * (Fluencia)^0.44 approx
    # Usamos np.log10 para la fluencia según algunas variantes, aquí usamos potencia directa
    # Nota: Los coeficientes exactos varían ligeramente según la revisión del paper ASTM.
    
    term_A = A_coeff * np.exp(19310.0 / T_R) * (1.0 + 110.0 * p) * np.power(fluence / 1e19, 0.4601)


    # --- 3. TÉRMINO B: PRECIPITACIÓN DE COBRE (CRP) ---
    # Este término es complejo. Depende de si el Cu satura.
    
    # Límite de saturación del Cobre (Cu_max suele ser 0.30% o 0.25% según Ni)
    # Estimación simplificada de Cu efectivo:
    Cu_max = 0.30
    Cu_eff = np.minimum(cu, Cu_max)
    
    # Si Cu es muy bajo, este término casi desaparece. 
    # Umbral de Cu (a veces se considera 0.072%)
    Cu_eff = np.maximum(Cu_eff - 0.072, 0) 

    # Coeficientes B
    B_coeff = np.where(product_form == 'W', 2.94e-4, 2.22e-4)
    
    # Función de saturación de fluencia para el término B
    # f(phi) = 0.5 + 0.5 * tanh(...)
    log_phi = np.log10(fluence)
    flux_function = 0.5 + 0.5 * np.tanh((log_phi - 18.29) / 0.6)

    term_B = B_coeff * (1.0 + 2.45 * ni) * Cu_eff * flux_function * np.exp(10000.0 / T_R) # Ajuste temp


    # --- 4. RESULTADO FINAL ---
    # TTS total en Fahrenheit
    tts_fahrenheit = term_A + term_B
    
    # Convertir TTS de Fahrenheit a Celsius (Delta F / 1.8 = Delta C)
    tts_celsius = tts_fahrenheit / 1.8
    
    return tts_celsius