from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model (đổi tên nếu bạn train model khác)
model = YOLO("runs/detect/train2/weights/best.pt")

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
    print("APP STARTED")
    app.run(debug=True)

