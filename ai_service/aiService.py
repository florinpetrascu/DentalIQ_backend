from ultralytics import YOLO
from domain.teeth import Teeth

import cv2
class AiService:
    def __init__(self, model1_path, model2_path):
          # Assuming you're using YOLO models
        self.model1 = YOLO(model1_path)  # Load the first model
        self.model2 = YOLO(model2_path)  # Load the second model

    def predict_with_model1(self, image_path):
        return self.model1.predict(source=image_path)

    def predict_with_model2(self, image_path):
        return self.model2.predict(source=image_path)

    def get_teeths(self, image):
        # Run the model on the input image
        result = self.predict_with_model1(image)

        teeth_list = []  # List to store Teeth objects

        # Extract bounding boxes and segmentation masks
        boxes = result[0].boxes
        masks = result[0].masks
        probs = result[0].probs

        if masks is not None:
            # Iterate over each detected object
            for i in range(len(masks.data)):
                # Get the polygon (contour) of the mask
                mask = masks.data[i].numpy()
                contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Choose the largest contour as the polygon (in case of multiple)
                if contours:
                    polygon = max(contours, key=cv2.contourArea).squeeze().tolist()
                else:
                    polygon = []

                # Get the class name (label)
                class_id = int(boxes.cls[i].item())  # Class ID
                #class_name = self.class_names[class_id]  # Map ID to name

                # Create a Teeth object and add it to the list
                teeth = Teeth(name=class_id, polygon=polygon, issues=[])
                teeth_list.append(teeth)

        return teeth_list

