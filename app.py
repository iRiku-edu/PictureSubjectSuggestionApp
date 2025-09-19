import os
from flask import Flask, request, render_template
from model import detect, predict

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = "./static/picture_image"

# 予測結果を受け渡すための関数を定義
results = None

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_user_files():
    if request.method == "POST":
        global results
        upload_file = request.files["upload_file"]
        img_path = os.path.join(UPLOAD_FOLDER, upload_file.filename)
        upload_file.save(img_path)
        results, detected_objects_info, output_path = detect(img_path)
        return render_template(
            "select.html", result=detected_objects_info, img_path=output_path
        )


@app.route("/select", methods=["GET", "POST"])
def select_candidate():
    global results
    if request.method == "POST":
        selected_candidate = request.form["selected_candidate"]
        present_subject, advice = predict(results, selected_candidate)
        return render_template("result.html", result=present_subject, advice=advice)


if __name__ == "__main__":
    app.run(debug=True)
