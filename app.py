from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image
import os

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train2", "weights", "best.pt")

model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # YOLO predict
            results = model(img_path, conf=0.25)

            # Save result image
            save_path = os.path.join(RESULT_FOLDER, file.filename)
            results[0].save(filename=save_path)

            result_image = save_path

    return render_template("index.html", result_image=result_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


