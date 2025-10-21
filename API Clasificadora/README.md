🧠 EyeDisease Classifier API

API desarrollada en Python + FastAPI + PyTorch para la clasificación automática de enfermedades oculares a partir de imágenes. El modelo fue entrenado con el dataset EyeDisease y utiliza una arquitectura basada en EfficientNet con capas personalizadas.

🚀 Características

Clasificación automática de imágenes oculares. Devuelve un JSON con las probabilidades para cada clase. Soporte para subida de archivos (multipart/form-data). Modelo optimizado para inferencia en CPU.

🧩 Tecnologías utilizadas

Python 3.12+ FastAPI (para la API REST) Uvicorn (servidor ASGI) PyTorch (red neuronal y carga del modelo) Torchvision (transformaciones de imágenes) Pillow (PIL) (procesamiento de imágenes)

⚙️ Instalación

Cloná el repositorio y accedé al directorio:

git clone https://github.com/ignaciop000/TP-IAA.git

cd API Clasificadora

Instalá las dependencias:

pip install -r requirements.txt

🧱 Archivos principales

api_eye_disease.py Código principal de la API modelo_cnn.pth Modelo entrenado (pesos) requirements.txt Lista de dependencias README.md Documentación del proyecto

▶️ Ejecución

Iniciá el servidor local con:

uvicorn api_eye_disease:app --reload

Por defecto, la API quedará disponible en:

http://127.0.0.1:8000

📤 Uso de la API Endpoint: /predict

Método: POST Tipo de contenido: multipart/form-data

Parámetro:

file: imagen ocular (.jpg, .png, etc.)

Ejemplo con curl: curl -X POST "http://127.0.0.1:8000/predict"
-F "file=@/ruta/a/imagen.jpg"

Alternativa con navegador, abrí:

http://127.0.0.1:8000/docs

Respuesta:

{ "predictions": [ { "class": "cataract", "probability": 0.0012410744093358517 }, { "class": "diabetic_retinopathy", "probability": 0.9314017295837402 }, { "class": "glaucoma", "probability": 0.06734415143728256 }, { "class": "normal", "probability": 0.00001296670325245941 } ] }

📊 Clases del modelo

Cataract Diabetic Retinopathy Glaucoma Normal

👨‍💻 Autor

Grupo B - IAA 2025 - FRBA - UTN

📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Podés usarlo libremente, citando al autor original.
