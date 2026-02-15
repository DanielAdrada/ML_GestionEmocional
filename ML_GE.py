import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔒 Reproducibilidad
np.random.seed(42)

# ==========================
# 1️⃣ EMOCIONES
# ==========================

emociones = [
    "Alegría",
    "Tristeza",
    "Ansiedad",
    "Miedo",
    "Enojo",
    "Frustración",
    "Confianza"
]

# ==========================
# 2️⃣ GENERADOR CONTROLADO SIN RUIDO FUERTE
# ==========================

def generar_fila_balanceada(emocion_objetivo):

    # Base baja con variabilidad real (0–1)
    respuestas = np.random.randint(0, 2, 21)

    estructura = {
        "Alegría": {
            "nucleo": [0,1,17,19],
            "sec": [16]
        },
        "Tristeza": {
            "nucleo": [8,10,13,18],
            "sec": [11]
        },
        "Ansiedad": {
            "nucleo": [2,3,9,11,14],
            "sec": [12]
        },
        "Miedo": {
            "nucleo": [4,7,12,15],
            "sec": [2]
        },
        "Enojo": {
            "nucleo": [5,6,7,15],
            "sec": [14]
        },
        "Frustración": {
            "nucleo": [6,14,15,17],
            "sec": [10]
        },
        "Confianza": {
            "nucleo": [16,17,19,20],
            "sec": [0]
        }
    }

    # Activar núcleo fuerte (2–3)
    for i in estructura[emocion_objetivo]["nucleo"]:
        respuestas[i] = np.random.randint(2,4)

    # Activar secundarias leve (1–2)
    for i in estructura[emocion_objetivo]["sec"]:
        respuestas[i] = np.random.randint(1,3)
    
    if np.random.rand() < 0.1:
        emocion_extra = np.random.choice(list(estructura.keys()))
        if emocion_extra != emocion_objetivo:
            for i in estructura[emocion_extra]["sec"]:
                respuestas[i] = np.random.randint(1,3)

    return list(respuestas.astype(int)) + [emocion_objetivo]

# ==========================
# 3️⃣ CREAR DATASET
# ==========================

dataset = []

for emocion in emociones:
    for _ in range(1000):
        dataset.append(generar_fila_balanceada(emocion))

columnas = [f"P{i}" for i in range(1,22)] + ["Emocion"]
df = pd.DataFrame(dataset, columns=columnas)

df.to_csv("dataset_GestionEmociones_BALANCEADO.csv", index=False)
print("✅ Dataset balanceado generado")

# ==========================
# 4️⃣ ENTRENAMIENTO
# ==========================

X = df.drop("Emocion", axis=1)
y = df["Emocion"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "modelo_GestionEmociones.pkl")
joblib.dump(encoder, "encoder_GestionEmociones.pkl")

print("✅ Modelo entrenado y guardado")

# ==========================
# 5️⃣ VALIDACIÓN
# ==========================

scores = cross_val_score(model, X, y_encoded, cv=5)
print("📊 Cross-validation promedio:", round(scores.mean(),4))

y_pred = model.predict(X_test)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Reporte por emoción:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
