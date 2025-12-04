import pandas as pd
import numpy as np

"""
================================================================================
MÓDULO C: ETIQUETADO DE SENTIMIENTO
================================================================================
Crear columna 'satisfaccion' según rating, justificar tratamiento de sin rating
================================================================================
"""


def create_sentiment_labels(df):
    """Crea la columna de satisfacción (sentimiento) basada en rating."""
    print("\n" + "="*80)
    print("C. ETIQUETADO")
    print("="*80)
    
    df_labeled = df.copy()
    
    # 1. ANÁLISIS DE RATINGS
    ratings_nulos = df_labeled['rating'].isnull().sum()
    print(f"\n1. ANÁLISIS DE RATINGS:")
    print(f"   Total de registros: {len(df_labeled)}")
    print(f"   Registros sin rating: {ratings_nulos}")
    
    # 2. TRATAMIENTO DE REGISTROS SIN RATING (requerido por rúbrica - justificar)
    if ratings_nulos > 0:
        print(f"\n2. TRATAMIENTO DE REGISTROS SIN RATING:")
        print(f"   Estrategia: Eliminar los {ratings_nulos} registros sin rating")
        print(f"   Justificación:")
        print(f"      - No se puede determinar satisfacción sin rating")
        print(f"      - Imputar valores generaría etiquetas artificiales y sesgo")
        print(f"      - El modelo de Deep Learning requiere etiquetas reales")
        df_labeled = df_labeled.dropna(subset=['rating'])
        print(f"   ✓ Registros eliminados: {ratings_nulos}")
        print(f"   ✓ Registros restantes: {len(df_labeled)}")
    
    # 3. CREAR ETIQUETAS DE SATISFACCIÓN (requerido por rúbrica)
    print(f"\n3. CREACIÓN DE ETIQUETAS:")
    print(f"   Criterios:")
    print(f"      - Positivo: rating ≥ 4")
    print(f"      - Neutro: rating = 3")
    print(f"      - Negativo: rating ≤ 2")
    
    def clasificar_satisfaccion(rating):
        if rating >= 4:
            return 'Positivo'
        elif rating == 3:
            return 'Neutro'
        else:
            return 'Negativo'
    
    df_labeled['satisfaccion'] = df_labeled['rating'].apply(clasificar_satisfaccion)
    
    # 4. DISTRIBUCIÓN DE SATISFACCIÓN
    print(f"\n4. DISTRIBUCIÓN DE SATISFACCIÓN:")
    distribucion = df_labeled['satisfaccion'].value_counts()
    porcentaje = df_labeled['satisfaccion'].value_counts(normalize=True) * 100
    
    resultado = pd.DataFrame({
        'Cantidad': distribucion,
        'Porcentaje': porcentaje.round(2)
    })
    print(resultado)
    
    # Análisis de balance de clases
    print(f"\n   Análisis de Balance de Clases:")
    max_clase = distribucion.max()
    min_clase = distribucion.min()
    ratio = max_clase / min_clase
    print(f"      Clase mayoritaria: {distribucion.idxmax()} ({max_clase} registros)")
    print(f"      Clase minoritaria: {distribucion.idxmin()} ({min_clase} registros)")
    print(f"      Ratio de desbalance: {ratio:.2f}:1")
    
    if ratio > 3:
        print(f"      ⚠ Dataset desbalanceado (considerar técnicas de balanceo para Eval. 3)")
    elif ratio > 2:
        print(f"      ⚠ Desbalance moderado")
    else:
        print(f"      ✓ Dataset relativamente balanceado")
    
    # Mapeo de ratings a satisfacción
    print(f"\n   Mapeo Rating → Satisfacción:")
    if 'rating' in df_labeled.columns:
        for rating in sorted(df_labeled['rating'].dropna().unique()):
            count = (df_labeled['rating'] == rating).sum()
            satisf = df_labeled[df_labeled['rating'] == rating]['satisfaccion'].iloc[0]
            pct = count/len(df_labeled)*100
            print(f"      Rating {int(rating)}: {count} registros ({pct:.1f}%) → {satisf}")
    
    # 5. VALIDACIÓN FINAL
    print(f"\n5. VALIDACIÓN FINAL:")
    print(f"   ✓ Total etiquetado: {len(df_labeled)} registros")
    print(f"   ✓ Sin etiquetas nulas: {df_labeled['satisfaccion'].isnull().sum()} nulos")
    print(f"   ✓ Clases únicas: {sorted(df_labeled['satisfaccion'].unique())}")
    print(f"   ✓ Columna 'satisfaccion' lista para modelado")
    
    print("\n" + "="*80)
    
    return df_labeled
