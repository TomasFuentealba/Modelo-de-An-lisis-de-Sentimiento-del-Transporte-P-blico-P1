import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos limpios
df = pd.read_csv('transporte_santiago_clean.csv')

# Configurar Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review_text'])

# Guardar tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convertir texto a secuencias
sequences = tokenizer.texts_to_sequences(df['review_text'])

# Aplicar padding
X = pad_sequences(sequences, maxlen=100)

# Codificar target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['satisfaccion'])

# Guardar label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Dividir en Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Guardar arrays
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preparación completada:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Clases: {label_encoder.classes_}")
