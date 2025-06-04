

# from ultralytics import YOLO
#
# # Încarcă modelul
# model = YOLO('teeth_model.pt')  # calea către modelul tău antrenat
#
# # Exportă în ONNX
# model.export(
#     format='onnx',
#     opset=12,               # versiunea ONNX. Recomandat >=12
#     imgsz=(640, 640),        # dimensiunea la care ai antrenat
#     dynamic=True,            # permite batch-size variabil
#     simplify=True,           # simplifică modelul
#     half=False               # fp32 (nu fp16) - dacă vrei fp16, pune True
# )