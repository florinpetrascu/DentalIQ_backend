
#
# from ultralytics import YOLO
#
# model = YOLO("controller/teeth_model.pt")
# model.export(format="onnx", dynamic=True)


import onnxruntime as ort
import cv2
import numpy as np

# 🔹 Încarcă modelul ONNX
model_path = "C://Users//40753//source//Facultate//Semestrul_5//ProiectareaProdInovative//DentalIQ_backend//controller//"  # Schimbă cu numele corect
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# 🔹 Încarcă imaginea
image_path = "C://Users//40753//source//Facultate//Semestrul_5//ProiectareaProdInovative//Radiografii//v19.jpg"  # Schimbă cu imaginea ta
image = cv2.imread(image_path)
input_size = 640  # Mărimea standard pentru YOLO

# 🔹 Preprocesează imaginea
img = cv2.resize(image, (input_size, input_size))
img = img / 255.0  # Normalizează la [0,1]
img = img.transpose(2, 0, 1)  # Rearanjează la (C, H, W)
img = np.expand_dims(img, axis=0).astype(np.float32)  # Adaugă batch dim

# 🔹 Rulează inferența
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: img})[0]

# 🔹 Procesează rezultatele

conf_threshold = 0.5
for det in outputs[0]:  # Format (x, y, w, h, conf, cls_id)
    x, y, w, h, conf, cls_id = det[:6]
    if conf > conf_threshold:
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"ID {int(cls_id)} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 🔹 Afișează imaginea finală
cv2.imshow("Predicție", image)
cv2.waitKey(0)
cv2.destroyAllWindows()