"""
================================================================================
PIPELINE DE PROCESAMIENTO DE DATOS - TRANSPORTE SANTIAGO
================================================================================

Evaluación N°2: Recolección y Preparación de Datos
Ponderación: 20%
Fecha Entrega: 27-11-2025

FLUJO DEL PIPELINE (según rúbrica):

A. EXPLORACIÓN INICIAL DEL DATASET
   └─ Registros, tipos, nulos, duplicados, inconsistencias
   
B. LIMPIEZA DE DATOS
   └─ Nulos, duplicados, normalización, tipos, datetime
   
C. ETIQUETADO DE SENTIMIENTO
   └─ Crear columna 'satisfaccion' basada en rating
   
D. VALIDACIÓN CRUZADA Y DIVISIÓN TRAIN/TEST
   └─ División 80/20 + K-Fold Cross Validation
   
E. GUARDAR Y EXPORTAR
   └─ Generar transporte_santiago_clean.csv

NOTA: El modelo de Deep Learning se implementará en la Evaluación N°3

================================================================================
"""

from importlib import import_module

# Importar funciones de los módulos
load_and_explore_data = import_module('01_data_loader').load_and_explore_data
clean_data = import_module('02_data_cleaning').clean_data
create_sentiment_labels = import_module('03_data_imputation').create_sentiment_labels
train_test_split_and_kfold = import_module('04_data_new_features').train_test_split_and_kfold
save_final_dataset = import_module('06_data_saving').save_final_dataset


def run_pipeline(input_file):
    """
    Ejecuta el pipeline completo de preparación de datos según la rúbrica de evaluación N°2.
    
    Este pipeline prepara el dataset limpio, estandarizado y etiquetado que servirá
    directamente como insumo para el entrenamiento del modelo de clasificación de 
    sentimiento en la Evaluación N°3.
    
    Args:
        input_file (str): Ruta del archivo CSV de entrada (resenas_transporte_santiago_1000.csv).
    
    Returns:
        dict: Resultados del pipeline con el DataFrame procesado y métricas de validación.
    """
    print("\n" + "=" * 80)
    print("PIPELINE: Preparación de Datos - Transporte Santiago")
    print("=" * 80)
    
    # A. EXPLORACIÓN
    df = load_and_explore_data(input_file)
    
    # B. LIMPIEZA
    df = clean_data(df)
    
    # C. ETIQUETADO
    df = create_sentiment_labels(df)
    
    # D. VALIDACIÓN
    X_train, X_test, y_train, y_test, kfold_results, df = train_test_split_and_kfold(df)
    
    # E. EXPORTACIÓN
    output_path = save_final_dataset(df)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETADO")
    print("=" * 80)
    
    print(f"✓ Archivo: {output_path}")
    print(f"✓ Registros: {len(df)} | Columnas: {len(df.columns)}")
    print(f"✓ CV Accuracy: {kfold_results['mean']:.4f} ± {kfold_results['std']:.4f}")
    print("=" * 80)
    
    return df, kfold_results


if __name__ == "__main__":
    INPUT_FILE = "resenas_transporte_santiago_1000.csv"
    run_pipeline(INPUT_FILE)
