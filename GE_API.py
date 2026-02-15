from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# =========================
# CARGA DE MODELO
# =========================

model = joblib.load("modelo_GestionEmociones.pkl")
encoder = joblib.load("encoder_GestionEmociones.pkl")

# =========================
# ENDPOINT PRINCIPAL
# =========================

@app.route("/predecir_emocion", methods=["POST"])
def predecir_emocion():

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No se recibió JSON válido"}), 400

    respuestas = data.get("respuestas")

    if not isinstance(respuestas, list) or len(respuestas) != 21:
        return jsonify({"error": "Se requieren exactamente 21 respuestas"}), 400

    # Convertir explícitamente a enteros
    try:
        respuestas = [int(x) for x in respuestas]
    except ValueError:
        return jsonify({"error": "Las respuestas deben ser números enteros"}), 400

    # Convertir a array numpy
    X = np.array(respuestas).reshape(1, -1)

    # =========================
    # PREDICCIÓN ML
    # =========================

    probabilidades = model.predict_proba(X)[0]
    emociones = encoder.classes_

    # Crear diccionario emoción -> porcentaje
    scores = {
        emociones[i]: round(float(probabilidades[i]) * 100, 2)
        for i in range(len(emociones))
    }

    # Ordenar emociones por probabilidad
    emociones_ordenadas = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    emocion_principal = emociones_ordenadas[0][0]
    emocion_secundaria = (
        emociones_ordenadas[1][0]
        if len(emociones_ordenadas) > 1
        else None
    )

    # =========================
    # RESPUESTA FINAL LIMPIA
    # =========================

    return jsonify({
        "emocion_principal": emocion_principal,
        "emocion_secundaria": emocion_secundaria,
        "scores": emociones_ordenadas
    })


if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False)
