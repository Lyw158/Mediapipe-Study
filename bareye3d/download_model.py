# download_model.py
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "models/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("下载 face_landmarker 模型...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
else:
    print("模型文件已存在")