from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# 모델 불러오기
model = tf.keras.models.load_model("cat_dog_classifier.keras")

@app.route("/", methods=["GET"])
def home():
    return "🐶🐱 강아지 vs 고양이 분류기 API 입니다. 이미지 업로드해서 예측하세요!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "🐱 고양이" if prediction < 0.5 else "🐶 강아지"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

