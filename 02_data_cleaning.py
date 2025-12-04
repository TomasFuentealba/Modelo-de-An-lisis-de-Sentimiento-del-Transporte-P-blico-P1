import pandas as pd
import numpy as np

"""
================================================================================
MÓDULO B: LIMPIEZA DE DATOS
================================================================================
Eliminar/imputar nulos, eliminar duplicados, normalizar categorías, 
convertir tipos de datos
================================================================================
"""


def clean_data(df):
    """Limpia el dataset según los requerimientos de la rúbrica."""
    print("\n" + "="*80)
    print("B. LIMPIEZA")
    print("="*80)
    
    df_clean = df.copy()
    inicial = len(df_clean)
    
    # 1. ELIMINAR DUPLICADOS (requerido por rúbrica)
    print(f"\n1. ELIMINACIÓN DE DUPLICADOS:")
    duplicados_antes = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"   Duplicados eliminados: {duplicados_antes} ({duplicados_antes/inicial*100:.2f}%)")
    print(f"   Registros restantes: {len(df_clean)}")
    
    # 2. TRATAMIENTO DE VALORES FALTANTES (requerido por rúbrica)
    print(f"\n2. TRATAMIENTO DE VALORES FALTANTES:")
    
    # Imputar valores numéricos con la mediana (robusto ante outliers)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        nulos = df_clean[col].isnull().sum()
        if nulos > 0:
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
            print(f"   {col}: {nulos} nulos imputados con mediana ({mediana:.2f})")
    
    # Imputar categorías con 'Desconocido'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'fecha':  # Excepto fecha
            nulos = df_clean[col].isnull().sum()
            if nulos > 0:
                df_clean[col] = df_clean[col].fillna('Desconocido')
                print(f"   {col}: {nulos} nulos imputados con 'Desconocido'")
    
    # 3. NORMALIZAR CATEGORÍAS (requerido por rúbrica)
    print(f"\n3. NORMALIZACIÓN DE CATEGORÍAS:")
    categorical_to_normalize = ['medio_transporte', 'empresa_bus', 'barrio']
    total_normalizaciones = 0
    for col in categorical_to_normalize:
        if col in df_clean.columns:
            antes = df_clean[col].nunique()
            # Mostrar ejemplos antes
            ejemplos_antes = df_clean[col].dropna().unique()[:3]
            print(f"\n   {col}:")
            print(f"      Antes: {antes} valores únicos, ej: {list(ejemplos_antes)}")
            
            df_clean[col] = df_clean[col].str.strip().str.title()
            despues = df_clean[col].nunique()
            cambio = antes - despues
            total_normalizaciones += cambio
            
            # Mostrar ejemplos después
            ejemplos_despues = df_clean[col].dropna().unique()[:3]
            print(f"      Después: {despues} valores únicos, ej: {list(ejemplos_despues)}")
            if cambio > 0:
                print(f"      ✓ {cambio} variantes unificadas")
    
    print(f"\n   Total de normalizaciones: {total_normalizaciones}")
    
    # 4. CONVERTIR FECHA A DATETIME (requerido por rúbrica)
    print(f"\n4. CONVERSIÓN DE TIPOS DE DATOS:")
    if 'fecha' in df_clean.columns:
        fechas_invalidas = pd.to_datetime(df_clean['fecha'], errors='coerce').isnull().sum() - df_clean['fecha'].isnull().sum()
        df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')
        print(f"   fecha: convertida a datetime ✓")
        if fechas_invalidas > 0:
            print(f"      ⚠ {fechas_invalidas} fechas inválidas convertidas a NaT")
        print(f"      Rango: {df_clean['fecha'].min()} a {df_clean['fecha'].max()}")
    
    # 5. CORREGIR TIPOS NUMÉRICOS (requerido por rúbrica)
    print(f"\n5. CORRECCIÓN DE TIPOS NUMÉRICOS:")
    numeric_corrections = {
        'rating': 'float64',
        'tiempo_espera_min': 'int64',
        'duracion_viaje_min': 'int64',
        'likes': 'int64',
        'respuestas': 'int64'
    }
    
    for col, dtype in numeric_corrections.items():
        if col in df_clean.columns:
            try:
                # Detectar valores no numéricos
                valores_invalidos = pd.to_numeric(df_clean[col], errors='coerce').isnull().sum() - df_clean[col].isnull().sum()
                
                if dtype == 'int64':
                    df_clean[col] = df_clean[col].fillna(0).astype(dtype)
                else:
                    df_clean[col] = df_clean[col].astype(dtype)
                
                print(f"   {col}: convertido a {dtype} ✓")
                if valores_invalidos > 0:
                    print(f"      ⚠ {valores_invalidos} valores no numéricos corregidos")
            except Exception as e:
                print(f"      ✗ Error: {e}")
    
    # 6. REPORTE POST-LIMPIEZA
    print(f"\n6. REPORTE POST-LIMPIEZA:")
    print(f"   Registros iniciales: {inicial}")
    print(f"   Registros finales: {len(df_clean)}")
    print(f"   Registros eliminados: {inicial - len(df_clean)} ({(inicial - len(df_clean))/inicial*100:.2f}%)")
    print(f"   Valores nulos restantes: {df_clean.isnull().sum().sum()}")
    print(f"   Duplicados restantes: {df_clean.duplicated().sum()}")
    print(f"   Calidad general: {(1 - df_clean.isnull().sum().sum()/(len(df_clean)*len(df_clean.columns)))*100:.2f}%")
    
    print(f"\n✓ Limpieza completada exitosamente")
    print("\n" + "="*80)
    
    return df_clean
