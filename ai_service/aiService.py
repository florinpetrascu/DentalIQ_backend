from ultralytics import YOLO
from domain.teeth import Teeth
from PIL import Image
from shapely.geometry import Polygon, box

import numpy as np
import cv2
class AiService:
    def __init__(self, model1_path, model2_path):
          # Assuming you're using YOLO models
        self.model1 = YOLO(model1_path)  # Load the first model
        self.model2 = YOLO(model2_path)  # Load the second model

        self.issue_labels = {
            0:  'Implant',
            1:  'Prosthetic restoration',
            2:  'Obturation',
            3:  'Endodontic treatment',
            4: ' Carious lesion',
            5:  'Bone resorbtion',
            6:  'Impacted tooth',
            7:  'Apical periodontitis',
            8:  'Root fragment',
            9:  'Furcation lesion',
            10: 'Apical surgery',
            11: 'Root resorption',
            12: 'Orthodontic device',
            13: 'Surgical device'
        }

    def teeth_predict(self, image_path):
        return self.model1.predict(source=image_path)

    def issue_predict(self, image_path):
        return self.model2.predict(source=image_path)

    def scale_coordinates(self,coords, original_shape, processed_shape):
        """
        Scale coordinates from processed image size to original image size.
        """
        scale_x = original_shape[1] / processed_shape[1]  # Width scale
        scale_y = original_shape[0] / processed_shape[0]  # Height scale

        scaled_coords = [
            [int(point[0] * scale_x), int(point[1] * scale_y)] for point in coords
        ]
        return scaled_coords

    from PIL import Image

    def resize_image(self, image, target_size=640):
        """
        Redimensionează imaginea astfel încât latura cea mai mare să fie 640, menținând proporțiile.

        Args:
            image (PIL.Image.Image): Imaginea de redimensionat.
            target_size (int): Dimensiunea dorită pentru latura cea mai mare.

        Returns:
            PIL.Image.Image: Imaginea redimensionată.
        """
        original_width, original_height = image.size

        # Calculează factorul de scalare
        if original_width > original_height:
            scale_factor = target_size / original_width
        else:
            scale_factor = target_size / original_height

        # Calculează dimensiunile noi
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Redimensionează imaginea
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def load_issues(self, teeth_list, image ):
        # Obține predicțiile pentru bounding boxes
        result = self.issue_predict(image)

        boxes = result[0].boxes

        # Convertim imaginea într-un format compatibil cu OpenCV
        image_cv = np.array(image)

        # Iterăm prin fiecare dinte
        for tooth in teeth_list:
            if len(tooth.polygon) > 2:
                tooth_polygon = np.array(tooth.polygon, dtype=np.int32)

                cv2.polylines(
                    image_cv, [tooth_polygon], isClosed=True, color=(255, 0, 0), thickness=2
                )


                cv2.putText(
                    image_cv,
                    str(tooth.name),
                    (int(tooth_polygon[0][0]), int(tooth_polygon[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=1,
                )

                tooth_polygon_shapely = Polygon(tooth.polygon)
                for i in range(len(boxes)):
                    # Obține coordonatele bounding box-ului
                    x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()

                    bbox = box(x_min, y_min, x_max, y_max)

                    # Desenează bounding box-ul
                    cv2.rectangle(
                        image_cv,
                        (int(x_min), int(y_min)),  # Convertește coordonatele în întregi
                        (int(x_max), int(y_max)),  # Convertește coordonatele în întregi
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    label = f"Class {self.issue_labels[int(boxes.cls[i].item())]}"
                    cv2.putText(
                        image_cv,
                        label,
                        (int(x_min),int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                    )

                    # Calculează aria intersecției
                    intersection_area = tooth_polygon_shapely.intersection(bbox).area
                    # Calculează aria poligonului dintelui
                    tooth_area = tooth_polygon_shapely.area

                    # Verifică dacă intersecția este mai mare de 20% din aria poligonului dintelui
                    if intersection_area > 0.2 * tooth_area:
                        tooth.addIssue(self.issue_labels[int(boxes.cls[i].item())])


        #cv2.imshow("Image with Bounding Boxes and Polygons", image_cv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



    def get_teeths(self, image):
        # Run the model on the input image
        # Obține dimensiunea originală a imaginii
        resized_image = self.resize_image(image, target_size=640)

        # Rulează modelul pentru detectarea dinților
        result = self.teeth_predict(image)

        result[0].show()

        teeth_list = []  # List to store Teeth objects

        # Extract bounding boxes and segmentation masks
        boxes = result[0].boxes
        masks = result[0].masks


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
                class_id = int(int(boxes.cls[i]))  # Class ID
                #print("classes",boxes.cls)
                #class_name = self.class_names[class_id]  # Map ID to name

                # Create a Teeth object and add it to the list
                teeth = Teeth(name=class_id, polygon=polygon, issues=[])
                teeth_list.append(teeth)


        self.load_issues(teeth_list,resized_image)

        for teeth in teeth_list:
            teeth.issues = list(set(teeth.issues))


        return teeth_list


#Theets  [Teeth(name=31, issues=['Prosthetic restoration']), Teeth(name=29, issues=['Prosthetic restoration']), Teeth(name=6, issues=['Prosthetic restoration']), Teeth(name=7, issues=['Endodontic treatment', 'Prosthetic restoration']), Teeth(name=24, issues=[]), Teeth(name=27, issues=[]), Teeth(name=28, issues=[]), Teeth(name=9, issues=[]), Teeth(name=21, issues=[]), Teeth(name=8, issues=[]), Teeth(name=20, issues=[]), Teeth(name=4, issues=['Prosthetic restoration']), Teeth(name=5, issues=['Prosthetic restoration']), Teeth(name=1, issues=['Impacted tooth']), Teeth(name=3, issues=['Prosthetic restoration']), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=23, issues=[]), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=22, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=18, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=15, issues=[]), Teeth(name=11, issues=['Prosthetic restoration']), Teeth(name=12, issues=['Prosthetic restoration']), Teeth(name=25, issues=[]), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=2, issues=['Prosthetic restoration']), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=26, issues=[]), Teeth(name=30, issues=['Prosthetic restoration'])]
