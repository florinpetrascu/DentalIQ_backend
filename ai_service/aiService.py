
from ultralytics import YOLO

from domain.issue import Issue
from domain.teeth import Teeth
from PIL import Image
from shapely.geometry import Polygon, Point , LineString
import random
from matplotlib import pyplot as plt
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

    import cv2
    import numpy as np
    import random
    from matplotlib import pyplot as plt

    def display_predictions(self, image, result):
        """
        Afișează bounding box-urile și etichetele pe imagine cu culori diferite pentru fiecare clasă.

        Args:
            image: Imaginea originală (format PIL sau numpy array).
            result: Rezultatele predicției YOLO (Ultralytics).
            issue_labels: Lista etichetelor claselor (ex. ["class1", "class2", ...]).
        """
        # Convertim imaginea într-un format OpenCV dacă e PIL
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Generăm culori unice pentru fiecare clasă



        boxes = result[0].boxes  # Obține toate bounding box-urile

        colors = {class_id: [random.randint(0, 255) for _ in range(3)] for class_id in range(len(boxes))}

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = map(int, boxes.xyxy[i].tolist())  # Coordonate bbox
            class_id = int(boxes.cls[i].item())  # Clasa prezisă
            confidence = boxes.conf[i].item()  # Confidența predicției

            # Numele clasei prezise
            label = f"{class_id} ({confidence:.2f})"
            color = colors[class_id]  # Culoarea clasei

            # Desenăm bounding box-ul pe imagine
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

            # Adăugăm eticheta deasupra bounding box-ului
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x, text_y = x_min, y_min - 5  # Poziția textului
            text_background = (x_min, y_min - text_size[1] - 5, x_min + text_size[0], y_min)

            # Desenăm un fundal pentru text (opțional)
            cv2.rectangle(image, (text_background[0], text_background[1]),
                          (text_background[2], text_background[3]), color, -1)

            # Adăugăm textul pe imagine
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness=font_thickness)

        # Afișăm imaginea folosind Matplotlib (nu blochează execuția)
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

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
        #result[0].show()
        #self.display_predictions(image,result)
        # Convertim imaginea într-un format compatibil cu OpenCV
        image_cv = np.array(image)
        print("teeth list len = ",len(teeth_list))



        # Iterăm prin fiecare dinte
        for tooth in teeth_list:
            if len(tooth.polygon) > 2:

                tooth_polygon_shapely = Polygon(tooth.polygon)



                for i in range(len(boxes)):
                    try:
                        class_id = int(boxes.cls[i].item())
                        x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()

                        if class_id == 12 or class_id == 1:

                            # Dimensiunea laturii pătratului
                            square_side = min(x_max - x_min, y_max - y_min)

                            # Centrul pătratului din stânga
                            left_center = (x_min + square_side / 2, (y_min + y_max) / 2)

                            # Centrul pătratului din dreapta
                            right_center = (x_max - square_side / 2, (y_min + y_max) / 2)



                            # Creează linia din cele două puncte
                            line = LineString([left_center, right_center])


                            # Verifică intersecția
                            if tooth_polygon_shapely.intersects(line) or tooth_polygon_shapely.contains(line):
                                issue = Issue(name=self.issue_labels[class_id])
                                tooth.addIssue(issue)
                                print(f"Issue PRR/SD adăugat: {issue.name}")



                        else:

                            # Calculăm centrul bbox-ului
                            bbox_center_x = (x_min + x_max) / 2
                            bbox_center_y = (y_min + y_max) / 2
                            bbox_center = Point(bbox_center_x, bbox_center_y)

                            # Verificăm dacă centrul bbox-ului este în interiorul poligonului dintelui
                            if tooth_polygon_shapely.contains(bbox_center):
                                issue = Issue(name=self.issue_labels[class_id])
                                tooth.addIssue(issue)
                                print(f"Issue adăugat: {issue.name}")



                    except Exception as e:
                        print(f"A apărut o eroare: {e}")

            print("Dinte procesat:", tooth.name)
        cv2.imwrite("processed_image.jpg", image_cv)
        print("load_issues finalizat")


        #cv2.imshow("Image with Bounding Boxes and Polygons", image_cv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



    def get_teeths(self, image):
        # Run the model on the input image
        # Obține dimensiunea originală a imaginii
        resized_image = self.resize_image(image, target_size=640)

        # Rulează modelul pentru detectarea dinților
        result = self.teeth_predict(image)

        #result[0].show()
        #self.display_predictions(image, result)
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

        print("get teeths")

        self.load_issues(teeth_list,resized_image)
        for teeth in teeth_list:
            # Elimină duplicatele din teeth.issues pe baza numelui (name)
            unique_issues = {issue.name: issue for issue in teeth.issues}.values()
            teeth.issues = list(unique_issues)

        print('get_teeths_finished')
        return teeth_list


#Theets  [Teeth(name=31, issues=['Prosthetic restoration']), Teeth(name=29, issues=['Prosthetic restoration']), Teeth(name=6, issues=['Prosthetic restoration']), Teeth(name=7, issues=['Endodontic treatment', 'Prosthetic restoration']), Teeth(name=24, issues=[]), Teeth(name=27, issues=[]), Teeth(name=28, issues=[]), Teeth(name=9, issues=[]), Teeth(name=21, issues=[]), Teeth(name=8, issues=[]), Teeth(name=20, issues=[]), Teeth(name=4, issues=['Prosthetic restoration']), Teeth(name=5, issues=['Prosthetic restoration']), Teeth(name=1, issues=['Impacted tooth']), Teeth(name=3, issues=['Prosthetic restoration']), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=23, issues=[]), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=22, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=18, issues=[]), Teeth(name=13, issues=['Prosthetic restoration']), Teeth(name=15, issues=[]), Teeth(name=11, issues=['Prosthetic restoration']), Teeth(name=12, issues=['Prosthetic restoration']), Teeth(name=25, issues=[]), Teeth(name=10, issues=['Prosthetic restoration']), Teeth(name=2, issues=['Prosthetic restoration']), Teeth(name=14, issues=['Prosthetic restoration']), Teeth(name=26, issues=[]), Teeth(name=30, issues=['Prosthetic restoration'])]
