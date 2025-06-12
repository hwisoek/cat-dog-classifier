import os
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# 환경에 따라 모델 경로 설정 (Colab vs Render)
if os.path.exists("/content/drive/MyDrive/cat_dog_classifier.keras"):
    model_path = "/content/drive/MyDrive/cat_dog_classifier.keras"  # Colab 환경
else:
    model_path = "cat_dog_classifier.keras"  # Render 환경

# 모델 불러오기
model = tf.keras.models.load_model(model_path)

# 이미지 전처리 함수
def preprocess_image(img):
    img = img.resize((150, 150))  # 모델 입력 크기 조정
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# HTML 페이지 렌더링 (파일 업로드 UI 추가)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "파일을 업로드하세요!", 400
        
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)

        prediction = model.predict(img_array, verbose=0)
        predicted_value = float(prediction[0][0])
        result = "🐱 고양이" if predicted_value < 0.5 else "🐶 강아지"

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


