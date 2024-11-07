from flask import Flask, request, jsonify, send_file
from PIL import Image
import io

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_img():
    if 'file' not in request.files:
        print('err 1')
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        print('err 2')
        return jsonify({"error": "No selected file"}), 400

    print('Img received')
    # TODO: Simulate img alteration then send back
    # TODO: it's just for highlighter
    image = Image.open(file).convert("L")
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    image.save(f"./destination/image.png")
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    # TODO: This should work on local as well as remote
    app.run(host='0.0.0.0', port=5000, debug=True)
