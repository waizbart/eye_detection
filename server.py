from flask import Flask, request, jsonify
from process_image import process_image
from PIL import Image
import numpy

app = Flask(__name__)

@app.route("/process_image", methods=["POST"])
def process_img():
    file = request.files['image']

    img = Image.open(file.stream)

    process = process_image(numpy.array(img))

    return jsonify({'msg': 'success', 'process_image': process})


if __name__ == "__main__":
    app.run(debug=True)