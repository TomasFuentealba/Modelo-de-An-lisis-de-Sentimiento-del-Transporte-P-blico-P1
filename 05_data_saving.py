import pandas as pd

"""
================================================================================
MÓDULO E: EXPORTACIÓN DEL DATASET FINAL
================================================================================
Guardado del CSV procesado y validación básica
================================================================================
"""


def save_final_dataset(df, output_path='transporte_santiago_clean.csv'):
    """Guarda el dataset procesado en un archivo CSV."""
    print("\n" + "="*80)
    print("E. EXPORTACIÓN")
    print("="*80)
    
    # 1. GUARDADO DEL ARCHIVO
    print(f"\n1. GUARDANDO ARCHIVO:")
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"   ✓ Dataset guardado: {output_path}")
    except Exception as e:
        print(f"   ✗ Error al guardar: {e}")
        return None
    
    # 2. VERIFICACIÓN DEL ARCHIVO
    print(f"\n2. VERIFICACIÓN DEL ARCHIVO:")
    
    # Comprobar existencia
    import os
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"   ✓ Archivo existe")
        print(f"   ✓ Tamaño: {file_size:.2f} KB")
    else:
        print(f"   ✗ Archivo no encontrado")
        return None
    
    # Recargar para verificar
    try:
        df_verificacion = pd.read_csv(output_path, encoding='utf-8')
        print(f"   ✓ Archivo legible con encoding UTF-8")
        print(f"   ✓ Registros: {len(df_verificacion)}")
        print(f"   ✓ Columnas: {len(df_verificacion.columns)}")
        
        # Verificar coincidencia
        if len(df_verificacion) == len(df):
            print(f"   ✓ Registros coinciden con el DataFrame original")
        else:
            print(f"   ⚠ Discrepancia: {len(df)} original vs {len(df_verificacion)} guardado")
        
        # 3. VALIDACIÓN DE CALIDAD (Requisitos de la rúbrica)
        print(f"\n3. VALIDACIÓN DE CALIDAD DEL DATASET FINAL:")
        
        # Sin nulos
        nulos_totales = df_verificacion.isnull().sum().sum()
        print(f"   Valores nulos: {nulos_totales}", "✓" if nulos_totales == 0 else "✗")
        
        # Sin duplicados
        duplicados = df_verificacion.duplicated().sum()
        print(f"   Duplicados: {duplicados}", "✓" if duplicados == 0 else "✗")
        
        # Categorías normalizadas
        if 'medio_transporte' in df_verificacion.columns:
            tiene_minusculas = df_verificacion['medio_transporte'].str.islower().any()
            tiene_mayusculas = df_verificacion['medio_transporte'].str.isupper().any()
            normalizado = not (tiene_minusculas or tiene_mayusculas)
            print(f"   Categorías normalizadas:", "✓" if normalizado else "⚠")
        
        # Columna satisfaccion presente y correcta
        if 'satisfaccion' in df_verificacion.columns:
            clases_correctas = set(df_verificacion['satisfaccion'].unique()) == {'Positivo', 'Neutro', 'Negativo'}
            print(f"   Columna 'satisfaccion': ✓")
            print(f"   Clases correctas (Positivo/Neutro/Negativo):", "✓" if clases_correctas else "✗")
            print(f"   Distribución:")
            for clase, count in df_verificacion['satisfaccion'].value_counts().items():
                print(f"      {clase}: {count} ({count/len(df_verificacion)*100:.1f}%)")
        else:
            print(f"   Columna 'satisfaccion': ✗ NO ENCONTRADA")
        
        # Tipos de datos correctos
        print(f"\n   Verificación de tipos de datos:")
        if 'fecha' in df_verificacion.columns:
            es_datetime = pd.api.types.is_datetime64_any_dtype(df_verificacion['fecha'])
            print(f"      fecha (datetime):", "✓" if es_datetime else "⚠ requiere conversión")
        
        numeric_cols = ['rating', 'tiempo_espera_min', 'duracion_viaje_min', 'likes', 'respuestas']
        for col in numeric_cols:
            if col in df_verificacion.columns:
                es_numerico = pd.api.types.is_numeric_dtype(df_verificacion[col])
                print(f"      {col} (numérico):", "✓" if es_numerico else "✗")
        
        # Mostrar primeras líneas
        print(f"\n4. VISTA PREVIA DEL ARCHIVO FINAL:")
        print(df_verificacion.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"   ✗ Error al verificar: {e}")
        return None
    
    print(f"\n✓ Exportación completada: {output_path}")
    print("="*80)
    
    return output_path
