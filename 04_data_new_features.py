import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

"""
================================================================================
MÓDULO D: VALIDACIÓN CRUZADA Y DIVISIÓN TRAIN/TEST
================================================================================
División 80/20 + K-Fold Cross Validation + Explicación
================================================================================
"""


def train_test_split_and_kfold(df):
    """Divide datos en train/test (80/20) y aplica K-Fold Cross Validation."""
    print("\n" + "="*80)
    print("D. VALIDACIÓN")
    print("="*80)
    
    # 1. PREPARACIÓN DE FEATURES
    print("\n1. PREPARACIÓN DE FEATURES:")
    
    X = df[['tiempo_espera_min', 'duracion_viaje_min', 'likes', 'respuestas']].copy()
    X.fillna(0, inplace=True)
    
    # Target
    le = LabelEncoder()
    y = le.fit_transform(df['satisfaccion'])
    
    print(f"   Features: {list(X.columns)}")
    print(f"   Dimensiones: {X.shape}")
    print(f"   Clases: {le.classes_}")
    
    # 2. DIVISIÓN TRAIN/TEST (80/20) - Requerido por rúbrica
    print("\n2. DIVISIÓN TRAIN/TEST (80/20):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verificar estratificación
    print(f"\n   Distribución estratificada:")
    for idx, clase in enumerate(le.classes_):
        train_pct = (y_train == idx).sum() / len(y_train) * 100
        test_pct = (y_test == idx).sum() / len(y_test) * 100
        print(f"      {clase}: Train={train_pct:.1f}%, Test={test_pct:.1f}%")
    
    # 3. K-FOLD CROSS VALIDATION - Requerido por rúbrica
    print("\n3. K-FOLD CROSS VALIDATION:")
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # EXPLICACIÓN DEL K-FOLD (requerido por rúbrica)
    print(f"\n   K-Fold (k={k}): Divide en {k} partes, rota validación, promedia resultados")
    print(f"   Razones: (1) Mejor uso de datos, (2) Estimación robusta,")
    print(f"            (3) Evita sesgo, (4) Detecta overfitting, (5) Estándar ML")
    
    # Aplicar K-Fold
    print(f"\n   Aplicando K-Fold con Logistic Regression:")
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    
    scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='accuracy')
    
    print(f"\n   Resultados por fold:")
    for i, score in enumerate(scores, 1):
        print(f"      Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n   Accuracy promedio: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"   Accuracy mínimo: {scores.min():.4f}")
    print(f"   Accuracy máximo: {scores.max():.4f}")
    
    # 4. EVALUACIÓN EN TEST SET
    print("\n4. EVALUACIÓN EN TEST SET:")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    test_accuracy = (y_pred == y_test).mean()
    print(f"   Accuracy en Test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Matriz de Confusión
    print(f"\n   Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   (Filas=Real, Columnas=Predicción)")
    print(f"\n   Clases: {le.classes_}")
    print(cm)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusión - Logistic Regression')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    try:
        plt.savefig('confusion_matrix_logistic.png', dpi=300, bbox_inches='tight')
        print(f"\n   ✓ Matriz guardada: confusion_matrix_logistic.png")
    except:
        print(f"\n   ⚠ No se pudo guardar la matriz")
    plt.close()
    
    # Classification Report
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Resultados
    kfold_results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'label_encoder': le
    }
    
    print("\n" + "="*80)
    
    return X_train, X_test, y_train, y_test, kfold_results, df
