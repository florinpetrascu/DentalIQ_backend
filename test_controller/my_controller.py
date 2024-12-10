from ai_service.aiService import AiService
from PIL import Image

teethModelPath = "C:\\Users\\40753\\source\\Facultate\\Semestrul_5\\MetodeInteligente\\Project\\Modele\\SegmentareDateMediciJson\\Yolo11l-seg_20_ep\\best (2).pt"
issueModelPath = "C:\\Users\\40753\\source\\Facultate\\Semestrul_5\\MetodeInteligente\\Project\\Modele\\DetectieBoliTratamente\\yolov8l_det_20ep\\best (4).pt"


try:
    # Open the image file
    image = Image.open("C:\\Users\\40753\\source\\Facultate\\Semestrul_5\\MetodeInteligente\\Project\\dental_yolo\\train\\images\\95.jpg")
    #image.show()  # Display the image
    print(f"Image format: {image.format}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

except Exception as e:
    print(f"Error loading image: {e}")

# Instantiate the AiService with the models
service = AiService(teethModelPath, issueModelPath)

# results = service.predict_with_model1(image)
#
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


theets = service.get_teeths(image)

print("Theets ",theets)