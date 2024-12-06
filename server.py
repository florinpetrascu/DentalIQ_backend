from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import image_prepare
import io
app = Flask(__name__)
CORS(app)  # Activează CORS

@app.route('/')
def index():
    return render_template('index.html')  # Servește pagina principală

@app.route('/upload', methods=['POST'])
def upload_img():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    print('Image received')
    # Convertim imaginea la alb-negru
    image = Image.open(file)
    # Procesăm imaginea folosind funcția ta `image_prepare.predict`
    processed_image = image_prepare.predict(image)  # Se presupune că returnează o imagine PIL

    # Convertim imaginea procesată în BytesIO pentru a o trimite ca răspuns
    img_io = io.BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Returnăm imaginea procesată
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

