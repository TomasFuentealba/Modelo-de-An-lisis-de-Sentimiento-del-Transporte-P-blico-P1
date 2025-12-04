import pandas as pd
import numpy as np

"""
================================================================================
MÓDULO A: EXPLORACIÓN INICIAL DEL DATASET
================================================================================
Identifica: cantidad de registros, tipos de datos, nulos, duplicados, 
inconsistencias en categorías
================================================================================
"""


def load_and_explore_data(file_path):
    """Carga y explora el dataset según criterios de la rúbrica."""
    print("\n" + "="*80)
    print("A. EXPLORACIÓN INICIAL")
    print("="*80)
    
    df = pd.read_csv(file_path)
    
    # 1. CANTIDAD DE REGISTROS (requerido por rúbrica)
    print(f"\n1. CANTIDAD DE REGISTROS:")
    print(f"   Total de filas: {len(df):,}")
    print(f"   Total de columnas: {len(df.columns)}")
    print(f"   Dimensiones: {df.shape}")
    
    # 2. TIPOS DE DATOS (requerido por rúbrica)
    print(f"\n2. TIPOS DE DATOS:")
    print(df.dtypes)
    
    # 3. PORCENTAJE DE NULOS POR COLUMNA (requerido por rúbrica)
    print(f"\n3. VALORES NULOS:")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)
    nulos_df = pd.DataFrame({
        'Columna': df.columns,
        'Nulos': nulos.values,
        'Porcentaje': porcentaje_nulos.values
    })
    print(nulos_df[nulos_df['Nulos'] > 0])
    print(f"\n   Total de valores nulos: {df.isnull().sum().sum()}")
    
    # 4. DUPLICADOS (requerido por rúbrica)
    print(f"\n4. DUPLICADOS:")
    duplicados = df.duplicated().sum()
    porcentaje_dup = (duplicados / len(df) * 100).round(2)
    print(f"   Registros duplicados: {duplicados} ({porcentaje_dup}%)")
    
    # 5. INCONSISTENCIAS EN CATEGORÍAS (requerido por rúbrica)
    print(f"\n5. INCONSISTENCIAS EN CATEGORÍAS:")
    
    # Analizar categorías principales
    categorical_cols = ['medio_transporte', 'empresa_bus', 'barrio']
    for col in categorical_cols:
        if col in df.columns:
            valores_unicos = df[col].nunique()
            print(f"\n   {col} (valores únicos: {valores_unicos}):")
            print(df[col].value_counts().head(10))
            
            # Detectar inconsistencias por mayúsculas/minúsculas
            if df[col].dtype == 'object':
                valores_lower = df[col].dropna().str.lower()
                inconsistencias = valores_unicos - valores_lower.nunique()
                if inconsistencias > 0:
                    print(f"   ⚠ {inconsistencias} valores con problemas de capitalización")
                    # Mostrar ejemplos
                    for val in df[col].dropna().unique()[:5]:
                        val_lower = str(val).lower()
                        similares = [v for v in df[col].dropna().unique() if str(v).lower() == val_lower]
                        if len(similares) > 1:
                            print(f"      Ejemplo: {similares}")
                            break
    
    # 6. ESTADÍSTICAS DESCRIPTIVAS (Variables Numéricas)
    print(f"\n6. ESTADÍSTICAS DESCRIPTIVAS:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
        
        # Análisis de outliers (método IQR)
        print(f"\n   Análisis de Outliers (método IQR):")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                print(f"      {col}: {outliers} outliers detectados ({outliers/len(df)*100:.1f}%)")
    
    # 7. PRIMERAS FILAS DEL DATASET
    print(f"\n7. PRIMERAS FILAS:")
    print(df.head(3))
    
    print("\n✓ Exploración completada")
    print("="*80)
    
    return df
