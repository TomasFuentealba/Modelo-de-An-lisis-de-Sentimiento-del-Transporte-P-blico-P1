import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

# ============================================================================
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ============================================================================
MAX_WORDS = 5000          # Tamaño del vocabulario
EMBEDDING_DIM = 128       # Dimensión de los vectores de embedding
MAX_LENGTH = 100          # Longitud máxima de las secuencias
LSTM_UNITS_1 = 128        # Unidades de la primera capa LSTM
LSTM_UNITS_2 = 64         # Unidades de la segunda capa LSTM
DENSE_UNITS = 64          # Unidades de la capa densa
DROPOUT_RATE = 0.3        # Tasa de dropout para capas LSTM
DROPOUT_DENSE = 0.5       # Tasa de dropout para capa densa
BATCH_SIZE = 32           # Tamaño del batch
EPOCHS = 20               # Número máximo de épocas
VALIDATION_SPLIT = 0.2    # Proporción de datos para validación

# ============================================================================
# CARGA DE DATOS PREPROCESADOS
# ============================================================================
print("[INFO] Cargando datos preprocesados...")
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"  - X_train: {X_train.shape}")
print(f"  - y_train: {y_train.shape}")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_test: {y_test.shape}")

# ============================================================================
# CONVERTIR A ONE-HOT ENCODING
# ============================================================================
print("\n[INFO] Convirtiendo etiquetas a one-hot encoding...")
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(f"  - Número de clases: {num_classes}")
print(f"  - y_train (one-hot): {y_train.shape}")
print(f"  - y_test (one-hot): {y_test.shape}")

# ============================================================================
# DIVISIÓN TRAIN/VALIDATION (SPLIT EXPLÍCITO)
# ============================================================================
print("\n[INFO] Creando conjunto de validación...")
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=VALIDATION_SPLIT,
    stratify=np.argmax(y_train, axis=1),  # Split estratificado por clase
    random_state=42
)

print(f"  - Train: {X_train_fit.shape[0]} muestras")
print(f"  - Validation: {X_val.shape[0]} muestras")
print(f"  - Test: {X_test.shape[0]} muestras")

# ============================================================================
# CONSTRUCCIÓN DEL MODELO DE DEEP LEARNING
# ============================================================================
print("\n[INFO] Construyendo arquitectura del modelo...")

model = Sequential()

# Capa 1: Embedding
model.add(Embedding(
    input_dim=MAX_WORDS,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_LENGTH,
    name='embedding_layer'
))

# Capa 2: Primera capa LSTM (con return_sequences=True)
model.add(LSTM(
    LSTM_UNITS_1,
    dropout=DROPOUT_RATE,
    recurrent_dropout=DROPOUT_RATE,
    return_sequences=True,  # ← OBLIGATORIO para apilar capas LSTM
    name='lstm_layer_1'
))

# Capa 3: Segunda capa LSTM
model.add(LSTM(
    LSTM_UNITS_2,
    dropout=DROPOUT_RATE,
    recurrent_dropout=DROPOUT_RATE,
    name='lstm_layer_2'
))

# Capa 4: Capa densa con regularización
model.add(Dense(
    DENSE_UNITS,
    activation='relu',
    name='dense_layer'
))
model.add(Dropout(DROPOUT_DENSE, name='dropout_layer'))

# Capa 5: Capa de salida
model.add(Dense(
    num_classes,
    activation='softmax',
    name='output_layer'
))

# ============================================================================
# COMPILACIÓN DEL MODELO
# ============================================================================
print("\n[INFO] Compilando modelo...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Para one-hot encoded labels
    metrics=['accuracy']
)

# Construir el modelo antes de mostrar arquitectura
model.build(input_shape=(None, MAX_LENGTH))

# Mostrar arquitectura del modelo
print("\n" + "="*80)
print("ARQUITECTURA DEL MODELO")
print("="*80)
model.summary()
print("="*80)

# Contar parámetros
total_params = model.count_params()
print(f"\n[INFO] Parámetros totales entrenables: {total_params:,}")

# ============================================================================
# CONFIGURACIÓN DE CALLBACKS
# ============================================================================
print("\n[INFO] Configurando callbacks...")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ============================================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n[INFO] Iniciando entrenamiento...\n")
history = model.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),  # Usar validación explícita
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

print("\n[INFO] Entrenamiento completado.")

# ============================================================================
# VISUALIZACIÓN DEL ENTRENAMIENTO
# ============================================================================
print("\n[INFO] Generando gráficos de entrenamiento...")

plt.figure(figsize=(14, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], marker='s', label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o', label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], marker='s', label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos_entrenamiento.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráficos guardados en 'graficos_entrenamiento.png'")
plt.close()

# ============================================================================
# EVALUACIÓN EN CONJUNTO DE PRUEBA
# ============================================================================
print("\n[INFO] Evaluando modelo en conjunto de prueba...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*80}")
print("RESULTADOS FINALES EN TEST SET")
print(f"{'='*80}")
print(f"  • Test Loss:     {test_loss:.4f}")
print(f"  • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*80}\n")

# ============================================================================
# MATRIZ DE CONFUSIÓN Y REPORTE DE CLASIFICACIÓN
# ============================================================================
print("[INFO] Generando matriz de confusión...")

# Obtener predicciones
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Cargar label encoder para nombres de clases
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
except FileNotFoundError:
    class_names = ['Clase 0', 'Clase 1', 'Clase 2']
    print("label_encoder.pkl no encontrado, usando nombres genéricos")

# Generar matriz de confusión
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusión - Deep Learning', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('confusion_matrix_dl.png', dpi=300, bbox_inches='tight')
print("Matriz de confusión guardada en 'confusion_matrix_dl.png'")
plt.close()

# Reporte de clasificación
print("\n[INFO] Reporte de clasificación:\n")
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report)

# Guardar reporte en archivo
with open('classification_report_dl.txt', 'w', encoding='utf-8') as f:
    f.write("REPORTE DE CLASIFICACIÓN - DEEP LEARNING\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)
print("Reporte guardado en 'classification_report_dl.txt'")

# ============================================================================
# GUARDADO DEL MODELO Y ARTEFACTOS
# ============================================================================
print("\n[INFO] Guardando modelo y artefactos...")

# Guardar modelo en formato H5
model.save('modelo_sentimiento_transporte.h5')
print("Modelo guardado en 'modelo_sentimiento_transporte.h5'")

# Guardar historial de entrenamiento
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Historial guardado en 'training_history.pkl'")

print("\n" + "="*80)
print("PIPELINE COMPLETADO EXITOSAMENTE")
print("="*80)
