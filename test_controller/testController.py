from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import io
import  numpy as np
import cv2
from ai_service.aiService import AiService  # Ensure this is your defined class
from service.service import Service
import base64

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing

# Paths to models
teethModelPath = "C:\\Users\\40753\\source\\Facultate\\Semestrul_5\\MetodeInteligente\\Project\\Modele\\SegmentareDateMediciJson\\Yolo11l-seg_20_ep\\best (2).pt"
issueModelPath = "C:\\Users\\40753\\source\\Facultate\\Semestrul_5\\MetodeInteligente\\Project\\Modele\\DetectieBoliTratamente\\yolov8l_det_20ep\\best (4).pt"

# Instantiate the AiService with the models
serviceAI = AiService(teethModelPath, issueModelPath)
service = Service(serviceAI)
@app.route('/')
def index():
    return render_template('index.html')  # Serve the main page


@app.route('/upload', methods=['POST'])
def upload_img():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the uploaded image
        image = Image.open(file).convert('RGB')
        original_image = np.array(image)
        original_height, original_width = original_image.shape[:2]

        # Process the image using get_teeths
        teeth_list = service.get_teeths(image)

        # Generate masks and overlay them
        overlay = original_image.copy()
        teeth_data = []
        for tooth in teeth_list:
            polygon = np.array(tooth.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], color=(255, 255, 255))  # White color
            teeth_data.append({
                "name": tooth.name,
                "polygon": tooth.polygon
            })

        alpha = 0.5
        combined_image = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

        # Convert the image for response
        final_image = Image.fromarray(combined_image.astype('uint8'))
        img_io = io.BytesIO()
        final_image.save(img_io, 'PNG')
        img_io.seek(0)

        # Encode the image as base64
        encoded_image = base64.b64encode(img_io.getvalue()).decode('utf-8')

        print("encoded_image ",encoded_image)
        print("theeth_data ",teeth_data)
        # Return the processed image and teeth data
        return jsonify({
            "image": encoded_image,  # Base64 encoded image
            "teeth": teeth_data
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# @app.route('/upload', methods=['POST'])
# def upload_img():
#     # Ensure a file is provided
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         # Open the uploaded image
#         image = Image.open(file).convert('RGB')  # Asigură-te că imaginea este RGB
#         print('Imagine procesată')
#
#         # Imaginea originală
#         original_image = np.array(image)  # Convertim imaginea originală la NumPy
#         original_height, original_width = original_image.shape[:2]
#
#         # Obține rezultatele de la model
#         results = service.predict_with_model1(image)
#
#         # Măștile segmentate
#         masks = results[0].masks.data.cpu().numpy()  # Obținem măștile ca NumPy
#         colors = np.random.randint(0, 255, (masks.shape[0], 3), dtype=np.uint8)  # Culori random pentru fiecare clasă
#
#         # Suprapunem măștile colorate peste imaginea originală
#         overlay = original_image.copy()
#         for i, mask in enumerate(masks):
#             # Redimensionăm masca pentru a se potrivi imaginii originale
#             resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
#             overlay[resized_mask > 0] = colors[i]  # Aplicăm culoarea fiecărei clase
#
#         # Combinăm imaginea originală cu segmentările colorate (transparență)
#         alpha = 0.5  # Gradul de transparență
#         combined_image = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
#
#         # Convertim imaginea combinată într-un obiect PIL
#         final_image = Image.fromarray(combined_image.astype('uint8'))
#
#         # Pregătim imaginea pentru răspuns
#         img_io = io.BytesIO()
#         final_image.save(img_io, 'PNG')
#         img_io.seek(0)
#
#         return send_file(img_io, mimetype='image/png')
#
#     except Exception as e:
#         # Handle any errors during processing
#         print(e)
#         return jsonify({"error": str(e)}), 500
